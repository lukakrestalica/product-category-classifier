"""Luka Krestalica — treniranje, validacija i čuvanje klasifikatora.

Pokretanje iz korena projekta: python train_model.py
Model se bira na validacionom skupu; test se koristi tek nakon izbora.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from product_features import TitleStats, title_key

ROOT = Path(__file__).resolve().parent
LABELS = ['CPUs', 'Digital Cameras', 'Dishwashers', 'Freezers', 'Fridge Freezers',
          'Fridges', 'Microwaves', 'Mobile Phones', 'TVs', 'Washing Machines']
EXAMPLES = [
    ('iphone 7 32gb gold,4,3,Apple iPhone 7 32GB', 'Mobile Phones'),
    ('olympus e m10 mark iii geh use silber', 'Digital Cameras'),
    ('kenwood k20mss15 solo', 'Microwaves'),
    ('bosch wap28390gb 8kg 1400 spin', 'Washing Machines'),
    ('bosch serie 4 kgv39vl31g', 'Fridge Freezers'),
    ('smeg sbs8004po', 'Fridge Freezers'),
]


def load_data(path):
    """Ukloni prazne ciljeve/naslove, ujednači oznake i spreči tačne duplikate."""
    raw = pd.read_csv(path, encoding='utf-8-sig')
    raw.columns = raw.columns.str.strip()
    if raw.columns.duplicated().any() or not {'Product Title', 'Category Label'}.issubset(raw):
        raise ValueError('CSV mora imati jedinstvene kolone Product Title i Category Label.')
    info = {'raw_rows': len(raw), 'missing_by_column': raw.isna().sum().to_dict(),
            'raw_categories': raw['Category Label'].fillna('<missing>').value_counts().to_dict()}
    df = raw[['Product Title', 'Category Label']].dropna().copy()
    info['missing_title_or_label'] = len(raw) - len(df)
    df.columns = ['title', 'category']
    df['title'] = df.title.astype(str).str.strip()
    label_map = {x.casefold(): x for x in LABELS}
    label_map.update({'fridge': 'Fridges', 'cpu': 'CPUs', 'mobile phone': 'Mobile Phones'})
    df['category'] = df.category.astype(str).str.strip().str.casefold().map(label_map)
    valid = df.title.ne('') & df.category.notna()
    info['empty_or_unknown'] = int((~valid).sum())
    df = df.loc[valid].copy()
    df['key'] = df.title.map(title_key)
    conflict = df.groupby('key').category.transform('nunique').gt(1)
    info['conflicting_rows'] = int(conflict.sum())
    df = df.loc[~conflict].copy()
    info['duplicates_removed'] = int(df.duplicated('key').sum())
    df = df.drop_duplicates('key').reset_index(drop=True)
    info['clean_rows'] = len(df)
    info['clean_categories'] = df.category.value_counts().to_dict()
    if set(df.category) != set(LABELS) or df.category.value_counts().min() < 10:
        raise ValueError('Potrebno je najmanje 10 čistih naslova u svakoj od 10 kategorija.')
    return raw, df, info


def make_model(name):
    """Sve transformacije su u Pipeline-u, pa se uče samo na treningu."""
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=30000)
    features = [('word', word)]
    if 'char' in name:
        features.append(('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                                min_df=2, max_features=40000, sublinear_tf=True)))
    if 'stats' in name:
        features.append(('stats', Pipeline([('extract', TitleStats()), ('scale', MaxAbsScaler())])))
    weights = {n: (0.15 if n == 'stats' else 1.0) for n, _ in features}
    transform = FeatureUnion(features, transformer_weights=weights)
    classifier = (MultinomialNB() if name == 'NB_word' else
                  LogisticRegression(max_iter=1500, random_state=42) if name == 'LR_word' else
                  LinearSVC(max_iter=5000, random_state=42))
    return Pipeline([('features', transform), ('classifier', classifier)])


def run_training(data_path=ROOT / 'data/products.csv', output_dir=ROOT):
    """Uporedi unapred određene modele, testiraj izbor, pa treniraj finalni model."""
    output_dir = Path(output_dir)
    reports = output_dir / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    models = output_dir / 'models'; models.mkdir(parents=True, exist_ok=True)
    raw, df, info = load_data(data_path)
    # 60% trening, 20% validacija, 20% test. Isti skupovi za sve kandidate.
    dev, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.category)
    train, val = train_test_split(dev, test_size=0.25, random_state=42, stratify=dev.category)
    assert set(train.key).isdisjoint(val.key) and set(dev.key).isdisjoint(test.key)
    names = ['NB_word', 'LR_word', 'SVC_word', 'SVC_word_char', 'SVC_word_char_stats']
    rows = []
    for name in names:
        start = time.perf_counter()
        model = make_model(name).fit(train.title, train.category)
        pred = model.predict(val.title)
        rows.append({'model': name, 'accuracy': accuracy_score(val.category, pred),
                     'macro_f1': f1_score(val.category, pred, average='macro'),
                     'weighted_f1': f1_score(val.category, pred, average='weighted'),
                     'seconds': time.perf_counter() - start})
        print(f"{name}: validation macro-F1={rows[-1]['macro_f1']:.4f}", flush=True)
    comparison = pd.DataFrame(rows)
    # Makro F1 daje jednak značaj svakoj kategoriji. Stabilni redosled rešava izjednačenje.
    best = comparison.sort_values(['macro_f1', 'accuracy'], ascending=False, kind='stable').iloc[0]['model']
    evaluated = make_model(best).fit(dev.title, dev.category)
    test_pred = evaluated.predict(test.title)
    report = classification_report(test.category, test_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(test.category, test_pred, labels=LABELS)
    # Finalni model uči na svim čistim naslovima tek nakon izdvojene evaluacije.
    final_model = make_model(best).fit(df.title, df.category)
    with (models / 'product_classifier.pkl').open('wb') as f:
        pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    samples = pd.DataFrame(EXAMPLES, columns=['title', 'expected'])
    samples['prediction'] = final_model.predict(samples.title)
    samples['correct'] = samples.expected.eq(samples.prediction)
    samples['present_in_dataset'] = samples.title.map(title_key).isin(df.key)
    # Primeri iz zadatka su funkcionalna demonstracija, ne dodatna nezavisna metrika.
    comparison.to_csv(reports / 'validation_metrics.csv', index=False)
    pd.DataFrame(report).T.to_csv(reports / 'test_classification_report.csv')
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(reports / 'test_confusion_matrix.csv')
    samples.to_csv(reports / 'example_predictions.csv', index=False)
    errors = test.loc[test.category.ne(test_pred), ['title', 'category']].copy()
    errors['predicted'] = test_pred[test.category.ne(test_pred)]
    errors.to_csv(reports / 'test_errors.csv', index=False)
    meta = {'author': 'Luka Krestalica', 'source_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
            'cleaning': info, 'splits': {'train': len(train), 'validation': len(val), 'test': len(test)},
            'selected_model': best, 'test_accuracy': accuracy_score(test.category, test_pred),
            'test_macro_f1': f1_score(test.category, test_pred, average='macro'),
            'test_weighted_f1': f1_score(test.category, test_pred, average='weighted'),
            'majority_baseline_accuracy': float(test.category.eq(dev.category.mode().iloc[0]).mean()),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
            'final_training_rows': len(df), 'test_evaluation_training_rows': len(dev)}
    (reports / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Selected: {best}; test accuracy={meta['test_accuracy']:.4f}; macro-F1={meta['test_macro_f1']:.4f}")
    return meta, comparison, pd.DataFrame(report).T, cm, samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/products.csv')
    args = parser.parse_args()
    run_training(args.data)
