"""Interaktivno predlaganje kategorije: python predict_category.py."""
from pathlib import Path
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'models/product_classifier.pkl')
    parser.add_argument('--title', help='Jednokratna predikcija umesto interaktivne petlje.')
    args = parser.parse_args()
    # Učitavati samo sopstvene/proverene pickle fajlove: mogu izvršavati Python kod.
    try:
        with args.model.open('rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        parser.error('Model nije pronađen. Prvo pokrenite python train_model.py.')

    def predict(title):
        title = title.strip()
        if not title:
            print('Naslov je prazan. Unesite naziv proizvoda.')
            return
        print('Predložena kategorija:', model.predict([title])[0])
        # Model bira samo među kategorijama prisutnim u trening skupu.
        word = model.named_steps['features'].transformer_list[0][1]
        if word.transform([title]).nnz == 0:
            print('Napomena: nema poznatih reči; proverite predlog ručno.')

    if args.title is not None:
        predict(args.title)
        return
    print('Unesite naslov proizvoda; /izlaz završava program.')
    while True:
        try:
            title = input('\nNaslov: ')
        except (EOFError, KeyboardInterrupt):
            print('\nKraj programa.')
            break
        if title.strip().casefold() == '/izlaz':
            print('Kraj programa.')
            break
        predict(title)


if __name__ == '__main__':
    main()
