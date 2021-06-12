from typing import List, Any

import mysql.connector
import csv


class Database:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self._cursor = self.conn.cursor()

    @property
    def cursor(self):
        return self._cursor

    def select_query_fetch_one(self, query, values=None) -> List[Any]:
        self._cursor.execute(query, values)
        return self._cursor.fetchone()

    def select_query_fetch_all(self, query, values=None) -> List[Any]:
        self._cursor.execute(query, values)
        return self._cursor.fetchall()


def write_to_file(file_name: str, data: List, header=None):
    with open(f"{file_name}.csv", 'w', encoding='windows-1250', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for item in data:
            writer.writerow(item)


def main():
    database = Database(host="localhost",
                        user="root",
                        password="nenad",
                        database="psz_projekat")

    broj_prodaja, broj_izdavanje = broj_nekretnina(database)
    write_to_file('broj_prodaja_izdavanje', [[broj_prodaja, broj_izdavanje]], ["Prodaja", "Izdavanje"])

    prodaja, izdavanje = dohvati_nekretnine(database)
    write_to_file('lista_prodaja', prodaja)
    write_to_file('lista_izdavanje', izdavanje)

    prodaja = nekretnine_po_gradovima_prodaja(database)
    write_to_file('lista_prodaja_po_gradovima', prodaja, ["Grad", "Broj"])

    stan_uk, stan_neuk, kuca_uk, kuca_neuk = broj_uknjizenih_nekretnina(database)
    write_to_file('broj_uknjizenih', [[stan_uk, stan_neuk, kuca_uk, kuca_neuk]],
                  ["Stan - uk", "Stan - neuk", "Kuca - uk", "Kuca - neuk"])

    kuce, stanovi = najskuplje_kuce_i_stanovi_prodaja(database)
    write_to_file('lista_najskuplje_kuce', kuce)
    write_to_file('lista_najskuplji_stanovi', stanovi)

    kuce, stanovi = najvece_kuce_i_stanovi(database)
    write_to_file('lista_najvece_kuce', kuce)
    write_to_file('lista_najveci_stanovi', stanovi)

    nekretnine = novogradnja(database)
    write_to_file('lista_novogradnja', nekretnine)

    nekretnine = najveci_broj_soba(database)
    write_to_file('lista_najveci_broj_soba', nekretnine)

    nekretnine = kuce_najveci_plac(database)
    write_to_file('lista_najveci_plac_kuce', nekretnine)

    nekretnine = stanovi_najveca_kvadratura(database)
    write_to_file('lista_najveca_kvadratura_stanovi', nekretnine)


def broj_nekretnina(database: Database) -> List[int]:
    query = "select count(*) from psz_projekat.nekretnina where tip_ponude = 'Prodaja'"
    broj_prodaja = database.select_query_fetch_one(query)[0]
    query = "select count(*) from psz_projekat.nekretnina where tip_ponude = 'Izdavanje'"
    broj_izdavanje = database.select_query_fetch_one(query)[0]

    return [broj_prodaja, broj_izdavanje]


def dohvati_nekretnine(database: Database) -> List[List]:
    query = "select * from psz_projekat.nekretnina where tip_ponude = 'Prodaja'"
    prodaja = database.select_query_fetch_all(query)
    query = "select * from psz_projekat.nekretnina where tip_ponude = 'Izdavanje'"
    izdavanje = database.select_query_fetch_all(query)

    return [prodaja, izdavanje]


def nekretnine_po_gradovima_prodaja(database: Database) -> List:
    query = "select grad, count(*) from psz_projekat.nekretnina " \
            "where tip_ponude = 'Prodaja' " \
            "group by grad " \
            "order by count(*) desc;"
    prodaja = database.select_query_fetch_all(query)

    return prodaja


def broj_uknjizenih_nekretnina(database: Database) -> List:
    query = "select count(*) from psz_projekat.nekretnina where uknjizenost = 'DA' and tip_nekretnine = 'Stan';"
    res0 = database.select_query_fetch_one(query)[0]

    query = "select count(*) from psz_projekat.nekretnina where uknjizenost = '' and tip_nekretnine = 'Stan';"
    res1 = database.select_query_fetch_one(query)[0]

    query = "select count(*) from psz_projekat.nekretnina where uknjizenost = 'DA' and tip_nekretnine = 'Kuća';"
    res2 = database.select_query_fetch_one(query)[0]

    query = "select count(*) from psz_projekat.nekretnina where uknjizenost = '' and tip_nekretnine = 'Kuća';"
    res3 = database.select_query_fetch_one(query)[0]

    return [res0, res1, res2, res3]


def najskuplje_kuce_i_stanovi_prodaja(database: Database) -> List[List]:
    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Kuća' and tip_ponude = 'Prodaja' " \
            "order by cena desc " \
            "limit 30;"
    kuce = database.select_query_fetch_all(query)

    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Stan' and tip_ponude = 'Prodaja' " \
            "order by cena desc " \
            "limit 30;"
    stanovi = database.select_query_fetch_all(query)

    return [kuce, stanovi]


def najvece_kuce_i_stanovi(database: Database) -> List[List]:
    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Kuća' and tip_ponude = 'Prodaja' " \
            "order by kvadratura desc " \
            "limit 100;"
    kuce = database.select_query_fetch_all(query)

    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Stan' and tip_ponude = 'Prodaja' " \
            "order by kvadratura desc " \
            "limit 100;"
    stanovi = database.select_query_fetch_all(query)

    return [kuce, stanovi]


def novogradnja(database: Database) -> List:
    query = "select * from psz_projekat.nekretnina " \
            "where tip_objekta = 'Novogradnja' " \
            "order by cena desc"
    nekretnine = database.select_query_fetch_all(query)

    return nekretnine


def najveci_broj_soba(database: Database) -> List:
    query = "select * from psz_projekat.nekretnina " \
            "order by broj_soba desc " \
            "limit 30;"
    nekretnine = database.select_query_fetch_all(query)

    return nekretnine


def kuce_najveci_plac(database: Database) -> List:
    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Kuća'" \
            "order by povrsina_placa desc " \
            "limit 30;"
    nekretnine = database.select_query_fetch_all(query)

    return nekretnine


def stanovi_najveca_kvadratura(database: Database) -> List:
    query = "select * from psz_projekat.nekretnina " \
            "where tip_nekretnine = 'Stan'" \
            "order by kvadratura desc " \
            "limit 30;"
    nekretnine = database.select_query_fetch_all(query)

    return nekretnine


if __name__ == "__main__":
    main()

