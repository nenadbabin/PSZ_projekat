from utility.helpers import load_all_data

from visualizer.visualizer_utility import najzastupljeniji_delovi_beograda, broj_stanova_po_kvadraturi, \
    nekretnine_po_dekadatama, odnos_nekratnina_na_podaju_i_za_iznajmljivanje, nekretnine_po_opsezima, \
    nekretnine_sa_parkingom


def main():
    data_frame = load_all_data()

    najzastupljeniji_delovi_beograda(data_frame)
    broj_stanova_po_kvadraturi(data_frame)
    nekretnine_po_dekadatama(data_frame)
    odnos_nekratnina_na_podaju_i_za_iznajmljivanje(data_frame)
    nekretnine_po_opsezima(data_frame)
    nekretnine_sa_parkingom(data_frame)


if __name__ == "__main__":
    main()
