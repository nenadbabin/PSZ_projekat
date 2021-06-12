import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_interpreter.main_data_interpreter import Database


# def addlabels_hbar(x,y):
#     for i in range(len(x)):
#         plt.text(y[i] + 5, i, y[i])


database = Database(host="localhost",
                        user="root",
                        password="nenad",
                        database="psz_projekat")

# a)
# 10 najzastupljenijih delova Beograda koji imaju najveći broj nekretnina u ponudi
# (i u sekciji za prodaju, i u sekciji za iznajmljivanje, zbirno).
query = "select * from psz_projekat.nekretnina"

database.cursor.execute(query)
data_frame = pd.DataFrame(database.cursor.fetchall())
data_frame.columns=['id', 'tip_ponude', 'tip_nekretnine',
                    'broj_soba', 'spratnost', 'sprat',
                    'povrsina_placa', 'grejanje', 'grad',
                    'lokacija', 'mikrolokacija', 'kvadratura',
                    'parking', 'uknjizenost', 'terasa',
                    'lift', 'tip_objekta', 'cena']

data = data_frame
maska = data['grad'] == 'Beograd'
data = data[maska]
data_grupisano = data[['lokacija']].groupby('lokacija')
data_agregirano = data_grupisano['lokacija'].agg(np.size).sort_values(ascending=False)
data_agregirano = data_agregirano.head(10)
# data_agregirano = data_agregirano.to_frame()  # type: pd.DataFrame
delovi_beograda = data_agregirano.index

prodaja = []
izdavanje = []
for deo_beograda in delovi_beograda:
    data = data_frame
    maska = (data['grad'] == 'Beograd') & (data['tip_ponude'] == 'Prodaja') & (data['lokacija'] == deo_beograda)
    data = data[maska]   # type: pd.DataFrame
    prodaja.append(data.shape[0])
    pass

for deo_beograda in delovi_beograda:
    data = data_frame
    maska = (data['grad'] == 'Beograd') & (data['tip_ponude'] == 'Izdavanje') & (data['lokacija'] == deo_beograda)
    data = data[maska]   # type: pd.DataFrame
    izdavanje.append(data.shape[0])
    pass

labels = delovi_beograda
y1 = data_agregirano
y2 = prodaja
y3 = izdavanje

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.barh(x - width, y1, width, label='Ukupno')
rects2 = ax.barh(x, y2, width, label='Prodaja')
rects3 = ax.barh(x + width, y3, width, label='Izdavanje')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Najveci broj nekretnina u Beogradu')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.set_figwidth(15)
fig.set_figheight(10)

plt.show()

# plt.figure(figsize = (15, 5))
# xpoints = np.array(data_frame['deo_grada'])
# ypoints = np.array(data_frame['broj'])
#
# plt.barh(xpoints, ypoints)
# addlabels_hbar(xpoints, ypoints)
# plt.show()


# b)
# Broj stanova za prodaju prema kvadraturi, u celoj Srbiji
# (do 35 kvadrata, 36-50, 51-65, 66-80, 81-95, 96-110, 111 kvadrata i više).

brojevi_stanova = []

data = data_frame
maska = (data['kvadratura'] <= 35) & (data['tip_nekretnine'] == 'Stan')
data = data[maska]
brojevi_stanova.append(data.shape[0])

leva_granica = 36
desna_granica = 50

for i in range(0, 5):
    data = data_frame
    maska = (data['kvadratura'] >= leva_granica) & (data['kvadratura'] <= desna_granica) & (data['tip_nekretnine'] == 'Stan')
    data = data[maska]
    brojevi_stanova.append(data.shape[0])
    leva_granica += 15
    desna_granica += 15

data = data_frame
maska = (data['kvadratura'] >= 111) & (data['tip_nekretnine'] == 'Stan')
data = data[maska]
brojevi_stanova.append(data.shape[0])

fig, ax = plt.subplots()
x = np.array(["<=35", "36-50", "51-65", "66-80", "81-95", "96-110", ">=111"])
y = np.array(brojevi_stanova)

bar = ax.bar(x, y)
ax.bar_label(bar, padding=3)
plt.show()


# c)
# Broj izgrađenih nekretnina po dekadama (1951-1960, 1961-1970, 1971-1980, 1981-1990, 1991-2000, 2001-2010,
# 2011-2020)1, a obuhvatiti i sekcije za prodaju i za iznajmljivanje.

ukupno = []
prodaja = []
izdavanje = []

data = data_frame
maska = (data['tip_objekta'] == 'Novogradnja')
data = data[maska]
ukupno.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == 'Stara gradnja')
data = data[maska]
ukupno.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == '')
data = data[maska]
ukupno.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == 'Novogradnja') & (data['tip_ponude'] == 'Prodaja')
data = data[maska]
prodaja.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == 'Stara gradnja') & (data['tip_ponude'] == 'Prodaja')
data = data[maska]
prodaja.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == '') & (data['tip_ponude'] == 'Prodaja')
data = data[maska]
prodaja.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == 'Novogradnja') & (data['tip_ponude'] == 'Izdavanje')
data = data[maska]
izdavanje.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == 'Stara gradnja') & (data['tip_ponude'] == 'Izdavanje')
data = data[maska]
izdavanje.append(data.shape[0])

data = data_frame
maska = (data['tip_objekta'] == '') & (data['tip_ponude'] == 'Izdavanje')
data = data[maska]
izdavanje.append(data.shape[0])

labels = ["Novogradnja", "Stara gradnja", "Nepoznato"]
y1 = ukupno
y2 = prodaja
y3 = izdavanje

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.barh(x - width, y1, width, label='Ukupno')
rects2 = ax.barh(x, y2, width, label='Prodaja')
rects3 = ax.barh(x + width, y3, width, label='Izdavanje')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Novogradnja/Stara gradnja')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.set_figwidth(15)
fig.set_figheight(10)

plt.show()


# d)
# Broj (i procentualni odnos) nekretnina koje se prodaju i nekretnina koje se iznajmljuju,
# za prvih 5 gradova sa najvećim brojem nekretnina (za svaki grad posebno prikazati
# grafikon BROJ_ZA_PRODAJU : BROJ_ZA_IZNAJMLJIVANJE).

y = np.array([35, 25, 25, 15])
labels = ["Prodaja", "Iznajmljivanje"]

data = data_frame
data_grupisano = data[['grad']].groupby('grad')
data_agregirano = data_grupisano['grad'].agg(np.size).sort_values(ascending=False)
data_agregirano = data_agregirano.head(5)
gradovi = data_agregirano.index

for grad in gradovi:
    brojevi = []
    data = data_frame
    maska = (data['grad'] == grad) & (data['tip_ponude'] == "Prodaja")
    data = data[maska]
    brojevi.append(data.shape[0])

    data = data_frame
    maska = (data['grad'] == grad) & (data['tip_ponude'] == "Izdavanje")
    data = data[maska]
    brojevi.append(data.shape[0])

    y = np.array(brojevi)
    mylabels = ["Prodaja", "Izdavanje"]

    p, tx, autotexts = plt.pie(y, labels=labels, autopct="")

    for i, a in enumerate(autotexts):
        a.set_text(f"{brojevi[i]} ({round((brojevi[i] / np.array(brojevi).sum()) * 100, 2)})%")

    plt.title(grad)
    plt.show()


# e)
# Broj (i procentualni odnos) svih nekretnina za prodaju, koje po ceni pripadaju jednom od sledećih opsega:
# * manje od 49 999 €,
# * između 50 000 i 99 999 €,
# * između 100 000 i 149 999 €,
# * između 150 000 € i 199 999 €,
# * 200 000 € ili više.

brojevi_nekretnina_na_prodaju = []

data = data_frame
maska = (data['tip_ponude'] == 'Prodaja') & (data['cena'] <= 49999)
data = data[maska]
brojevi_nekretnina_na_prodaju.append(data.shape[0])

leva_granica = 50000
desna_granica = 99999

for i in range(0, 3):
    data = data_frame
    maska = (data['cena'] >= leva_granica) & (data['cena'] <= desna_granica) & (data['tip_ponude'] == 'Prodaja')
    data = data[maska]
    brojevi_nekretnina_na_prodaju.append(data.shape[0])
    leva_granica += 50000
    desna_granica += 50000

data = data_frame
maska = (data['cena'] >= 200000) & (data['tip_ponude'] == 'Prodaja')
data = data[maska]
brojevi_nekretnina_na_prodaju.append(data.shape[0])

y = np.array(brojevi_nekretnina_na_prodaju)
labels = ["<=49999", "50000-99999", "100000-149999", "150000-199999", ">=200000"]

p, tx, autotexts = plt.pie(y, labels=labels, autopct="")

for i, a in enumerate(autotexts):
    a.set_text(f"{brojevi_nekretnina_na_prodaju[i]} ({round((brojevi_nekretnina_na_prodaju[i] / np.array(brojevi_nekretnina_na_prodaju).sum()) * 100, 2)})%")

plt.title("Nekretnine na prodaju po opsezima cena")
plt.show()


# f)
# Broj nekretnina za prodaju koje imaju parking,
# u odnosu na ukupan broj nekretnina za prodaju (samo za Beograd).

data = data_frame
maska = (data['tip_ponude'] == 'Prodaja') & (data['grad'] == 'Beograd')
data = data[maska]
ukupan_broj_nekretnina = data.shape[0]

maska = (data['parking'] == 'DA')
data = data[maska]
nekretnine_sa_parkingom = data.shape[0]

nekretnine_parking = [nekretnine_sa_parkingom, ukupan_broj_nekretnina - nekretnine_sa_parkingom]
y = np.array(nekretnine_parking)
labels = ["Sa parkingom", "Bez parkinga"]

p, tx, autotexts = plt.pie(y, labels=labels, autopct="")

for i, a in enumerate(autotexts):
    a.set_text(f"{nekretnine_parking[i]} ({round((nekretnine_parking[i] / np.array(nekretnine_parking).sum()) * 100, 2)})%")

plt.title("Nekretnine na prodaju u Beogradu sa i bez parkinga")
plt.show()
