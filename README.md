# Predikcija kategorije proizvoda iz naslova

**Luka Krestalica — Task 3**

Model predlaže jednu od deset kategorija na osnovu naslova proizvoda. Projekat sadrži izvorni skup, izvršenu analizu, kod za ponovno treniranje, sačuvani model i interaktivni unos.

## Pokretanje

Preuzeti repozitorijum preko **Code → Download ZIP**, raspakovati ga i otvoriti terminal u njegovom folderu. Koristiti **Python 3.12**. Testirano sa verzijom 3.12.14.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python predict_category.py
```

Na Linuxu/macOS-u koristiti `.venv/bin/python` umesto `.venv\Scripts\python`.

Uneti naslov, na primer `kenwood k20mss15 solo`. Program ispisuje predloženu kategoriju i prihvata sledeći naslov. `/izlaz`, Ctrl+C ili kraj ulaza završavaju program. Prazan unos se odbija.

Jednokratna predikcija:

```powershell
.venv\Scripts\python predict_category.py --title "bosch serie 4 kgv39vl31g"
```

Ponovno treniranje i otvaranje sveske:

```powershell
.venv\Scripts\python train_model.py
.venv\Scripts\python -m jupyterlab product_category_analysis.ipynb
```

U svesci izabrati **Restart Kernel and Run All**. Sveska koristi isti kod kao trening skript i ponovo pravi model i izveštaje. Pokrenuti je iz korena projekta. `train_model.py --data putanja/do/products.csv` omogućava drugi ulaz istog formata.

## Organizacija

```text
data/products.csv                   Izvorni CSV iz zadatka
product_category_analysis.ipynb      Izvršena analiza i eksperimenti
train_model.py                      Čišćenje, poređenje, evaluacija i čuvanje
product_features.py                 Numeričke osobine naslova
predict_category.py                 Interaktivno učitavanje i predikcija
models/product_classifier.pkl       Kompletan trenirani Pipeline
reports/                            Metrike, matrica, greške, primeri i grafikoni
requirements.txt                    Biblioteke
```

`product_features.py` mora ostati uz skriptove zbog učitavanja sopstvene klase iz pickle fajla. Model se učitava bez ponovnog treniranja i bez interneta. Instalacija biblioteka zahteva internet. Učitavati samo sopstvene ili proverene `.pkl` fajlove, jer pickle može izvršiti kod. Za postojeći model koristiti navedene verzije biblioteka.

## Podaci i priprema

Izvor je dostavljeni **IMLP6_TASK_03-products.csv**, kopiran bez izmene u `data/products.csv`. Sadrži **35.311 redova i osam kolona**. Izvorni CSV je sačuvan radi reprodukcije; ovaj projekat ne dodeljuje novu licencu izvornom skupu.

- Uklonjeno je 215 redova bez naslova ili ciljne oznake.
- Standardizovane su oznake `CPU → CPUs`, `fridge → Fridges` i `Mobile Phone → Mobile Phones`.
- Uklonjena su četiri reda sa protivrečnim oznakama istog normalizovanog naslova.
- Uklonjeno je 4.270 ponovljenih naslova, nakon zanemarivanja veličine slova i višestrukih razmaka.
- Za model ostaje **30.822 naslova u deset kategorija**.

Prazne vrednosti u ostalim kolonama prijavljene su u svesci, ali ne utiču na trening: jedini ulaz je naslov. ID, prodavac, pregledi, ocena i datum nisu dostupni pri interaktivnom unosu i ne koriste se kao karakteristike.

## Eksperimenti i izbor

Stratifikovana podela sa semenom 42: **18.492 trening**, **6.165 validacija**, **6.165 test**. Isti naslovi ne prelaze između grupa. Test se ne koristi za izbor algoritma ili karakteristika.

| Kandidat | Validaciona tačnost | Validacioni makro F1 |
|---|---:|---:|
| Naive Bayes, reči | 93,43% | 0,933762 |
| Logistic Regression, reči | 95,02% | 0,951251 |
| Linear SVC, reči | 95,94% | 0,960454 |
| Linear SVC, reči + delovi reči | 98,43% | 0,983899 |
| Linear SVC, reči + delovi reči + numeričke osobine | 98,43% | **0,983943** |

Makro F1 je primarni kriterijum jer svim kategorijama daje isti značaj. Izabran je poslednji kandidat. Reči i parovi reči beleže brendove i izraze, a znakovni segmenti dužine 3–5 pomažu kod šifara i varijacija. Numeričke osobine obuhvataju dužinu naslova, broj reči, cifara, specijalnih znakova i velikih skraćenica, kao i dužinu najduže reči.

Najveći napredak dolazi od znakovnih segmenata. Numeričke osobine donose vrlo mali pomak makro F1 (oko 0,000044), bez promene tačnosti; to nije dokaz stabilnog poboljšanja na drugim podelama. TF-IDF, skaler i klasifikator uče zajedno unutar Pipeline-a, bez učenja na test tekstovima.

## Rezultat na izdvojenom testu

Izabrana konfiguracija ponovo je trenirana na treningu i validaciji (24.657 naslova), pa ocenjena na testu:

- **Accuracy: 98,77%**
- **Makro F1: 0,987665**
- **Ponderisani F1: 0,987695**
- **76 grešaka od 6.165 test naslova**
- Većinska referenca: 15,60% tačnosti.

Preciznost, odziv i F1 po kategoriji, matrica zabune i primeri grešaka nalaze se u svesci i `reports/`. Nakon evaluacije finalni `.pkl` treniran je na svih 30.822 naslova; gornje test metrike odnose se na model treniran na izdvojenih 80%, a ne na nezavisno testiranje završnog refita.

## Primeri iz zadatka i ograničenja

Pet od šest navedenih primera daje očekivanu kategoriju. `smeg sbs8004po` daje **Dishwashers**, dok zadatak očekuje **Fridge Freezers**. Greška je zadržana i dokumentovana; nema ručno zadatog pravila koje bi je prikrilo. Dva zadatka primera već postoje u skupu, pa rezultat 5/6 nije nezavisna procena kvaliteta.

Kratak naslov sa brendom i nepoznatom šifrom može biti nejasan. Model uvek bira među poznatih deset kategorija; novi tip proizvoda ne može pouzdano označiti kao nepoznat. Kod naslova bez poznatih reči prikazuje se napomena da se predlog proveri ručno. Model ne daje kalibrisanu verovatnoću.

Slični naslovi različitih varijanti istog proizvoda mogu ostati u različitim grupama i nakon uklanjanja identičnih tekstova. Za sledeću verziju potrebna je evaluacija po familijama proizvoda ili vremenu, više validiranih kratkih naslova i provera stabilnosti dodatnih karakteristika kroz više podela.

## Dokumentacija

- [scikit-learn: Pipeline i FeatureUnion](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn: čuvanje i učitavanje modela](https://scikit-learn.org/stable/model_persistence.html)

Instruktoru se predaje link javnog GitHub repozitorijuma.
