import json
import re
from typing import List

from scrapy import Spider, Request, Selector
from scrapy.http import HtmlResponse

from web_crawler.items import WebCrawlerItem


class HaloOglasiSpider(Spider):
    name: str = 'halo_oglasi'
    BASE_URL: str = 'https://www.halooglasi.com'

    def start_requests(self):
        urls: List[str] = [
            'https://www.halooglasi.com/nekretnine/prodaja-kuca?page=1',
            'https://www.halooglasi.com/nekretnine/izdavanje-kuca?page=1',
            'https://www.halooglasi.com/nekretnine/prodaja-stanova?page=1',
            'https://www.halooglasi.com/nekretnine/izdavanje-stanova?page=1'
        ]
        for url in urls:
            yield Request(url=url, callback=self.parse)

    def parse(self, response: HtmlResponse):
        """
        Example of product-title css class:
        <h3 class="product-title">
            <a href="/nekretnine/prodaja-kuca/extra-extra-lux-kuca--bez-provizije/5425636301903?kid=4">
                EXTRA EXTRA LUX KUĆA- BEZ PROVIZIJE
            </a>
        </h3>
        :param response: scrapy.http.Response
        :return:
        """
        product_relative_urls: List[str] = response.css(".product-title a::attr(href)").getall()

        for product_relative_url in product_relative_urls:
            product_absolute_url: str = self.BASE_URL + product_relative_url
            yield Request(product_absolute_url, callback=self.parse_product)

        if len(product_relative_urls):
            response_split = response.url.split("?page=", 1)
            current_page = int(response_split[1])
            next_page = current_page + 1
            yield response.follow(f"{response_split[0]}?page={next_page}", callback=self.parse)

    def parse_product(self, response: HtmlResponse):
        """
        QuidditaEnvironment.CurrentClassified={
        "Id":"5425635785546",
        "AdKindId":"4",
        "IsPromoted":false,
        "StateId":101,
        "AdvertiserId":"5849025",
        "Title":"Predivna kuca u oazi mira, moderno opremljena",
        "TextHtml":"Prodajemo porodicnu kucu potpuno opremljenu.Kuca je gradjena od najboljih materijala, sa predivnom
        terasom duz cele kuce koja je prirodnim kamenom poplocana. Od prostorija ima: ulaz, hodnik, &nbsp;veliko
        kupatilo sa prirodnom ventulacijom, kuhinja sa ostavom i prirodnom ventilacijom,trepezarija,dnevni boravak,
        2 komforne spavace sobe.Plac ravan 20 ari uredjen sa 3 parking mesta. U skolu kuce je i izlivena ploca za jos
        85kvadrata gradnje. Oaza mira i vazdusna banja na samo 22km od centra Beograda,3km od Ibarske magistrale,
        prodavnica i autobuska stanica na 500m. Za svaku preporuku! Zainteresovani se mogu javiti na 0638100200","
        Email":"True",
        "ValidFrom":"2021-06-03T16:43:05.927Z",
        "ValidTo":"2021-06-18T16:43:05.927Z",
        "GeoLocationRPT":"44.631928,20.453906",
        "ImageURLs":["/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166823.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166819.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166820.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166821.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166822.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166824.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166825.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166826.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166827.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166828.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166829.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166830.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166831.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792166832.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792167237.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792167238.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792167239.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792167240.jpg","/slike/oglasi/Thumbs/200727/l/predivna-kuca-u-oazi-mira-moderno-opremljena-5425635785546-71792167241.jpg"],
        "ImageTexts":["Staza do kuce","Put od kapije do kuce sa 3 parking mesta","Dvoriste","","Dvoriste","Dvoriste","Ulaz u kucu","Ulaz u kucu","Hodnik ","Trepezarija i dnevni boravak","Trepezarija","Kuhinja","Dnevna soba","Hodnik ka kupatilu","Spavaca soba","Spavaca soba","Spavaca soba","Spavaca soba","Kupatilo"],
        "CategoryIds":[1,2,2001,24],
        "CategoryNames":["Nekretnine","Stambeni prostor","Prodaja","Kuća"],
        "AdvertiserLogoUrl":null,
        "VideoUrl":null,
        "EnclosureFilePath":null,
        "DoNotShowContactButton":null,
        "ContactButtonLink":null,
        "OtherFields":
        {
        "broj_soba_s":"3.0",
        "spratnost_s":"prizemna",
        "povrsina_placa_d":20.0,
        "grejanje_s":"CG",
        "grad_s":"Beograd",
        "lokacija_s":"Opština Barajevo",
        "mikrolokacija_s":"Lipovica",
        "kvadratura_d":120.0,
        "oglasivac_nekretnine_s":"Vlasnik",
        "tip_nekretnine_s":"Kuća",
        "ulica_t":"Beogradska",
        "cena_d":155000.0,
        "dodatno_ss":["Uknjižen"],
        "ostalo_ss":["Garaža","Internet","Kanalizacija","KATV","Klima","Pomoćni objekti","Struja","Terasa","Voda"],
        "stanje_objekta_s":"Lux",
        "tip_objekta_s":"Novogradnja",
        "vrsta_objekta_s":"cela kuća",
        "mesecne_rezije_d":60.0,
        "broj_soba_id_l":404,
        "spratnost_id_l":472,
        "grejanje_id_l":1542,
        "grad_id_l":35112,
        "lokacija_id_l":528336,
        "mikrolokacija_id_l":528348,
        "oglasivac_nekretnine_id_l":387237,
        "tip_nekretnine_id_l":8100001,
        "dodatno_id_ls":[12000004],
        "ostalo_id_ls":[12100016,12100012,12100005,12100011,12100002,12100022,12100006,12100001,12100004],
        "stanje_objekta_id_l":11950001,
        "tip_objekta_id_l":387235,
        "vrsta_objekta_id_l":583,
        "broj_soba_order_i":7,
        "povrsina_placa_d_unit_s":"ar",
        "kvadratura_d_unit_s":"m2",
        "cena_d_unit_s":"EUR",
        "mesecne_rezije_d_unit_s":"EUR",
        "defaultunit_povrsina_placa_d":20.0,
        "defaultunit_kvadratura_d":120.0,
        "defaultunit_cena_d":155000.0,
        "defaultunit_mesecne_rezije_d":60.0,
        "_version_":1701557196351864832},
        "IsVerificationPending":false,
        "VerificationStateId":2,
        "TotalViews":18139,
        "IsOwnedByCurrentUser":false,
        "UseRaiffeisenCreditCalculator":false,
        "CreditInstalment":null,
        "CreditTotalAmount":null,
        "UseIntesaCreditCalculatorF":false,
        "UseOtpCreditCalculator":false,
        "UseSberbankCreditCalculator":false,
        "UseRaiffeisenCreditCalculatorNew":false,
        "RaiffeisenCreditCalculatorNewParentCategoryUrl":null,
        "IsInterestingInternal":false,
        "IsInterestingExternal":false,
        "RelativeUrl":"/nekretnine/prodaja-kuca/predivna-kuca-u-oazi-mira-moderno-opremljena/5425635785546?kid=4",
        "HasAutomaticRenewal":true,
        "ValidToProlonged":null,
        "ExpiresWithin48Hours":false,
        "AveragePriceBySurfaceValue":"823 €/m<sup>2</sup>",
        "AveragePriceBySurfaceLink":"/uporedne-cene-nekretnina?fromDate=1.5.2020.&toDate=1.5.2021.&categoryId=24&locations=528348&numOfRooms=404"}; for (var i in QuidditaEnvironment.CurrentClassified.OtherFields) { QuidditaEnvironment.CurrentClassified[i] = QuidditaEnvironment.CurrentClassified.OtherFields[i];
        };
        :param response:
        :return:
        """

        pattern = re.compile('QuidditaEnvironment.CurrentClassified=(.*?);')
        script = response.xpath("//script[contains(., 'QuidditaEnvironment.CurrentClassified=')]/text()")
        data = script.re(pattern)[0]
        data_obj = json.loads(data)

        item = WebCrawlerItem()

        item['tip_ponude'] = data_obj['CategoryNames'][2]

        item['tip_nekretnine'] = data_obj['OtherFields']['tip_nekretnine_s']

        if data_obj['OtherFields'].get('broj_soba_s'):
            item['broj_soba'] = data_obj['OtherFields'].get('broj_soba_s')
        else:
            item['broj_soba'] = 0
        if '+' in item['broj_soba']:
            item['broj_soba'] = '5.0'

        if data_obj['OtherFields'].get('sprat_od_s'):
            item['spratnost'] = int(data_obj['OtherFields'].get('sprat_od_s'))
        else:
            item['spratnost'] = 0

        if data_obj['OtherFields'].get('defaultunit_povrsina_placa_d'):
            item['povrsina_placa'] = float(data_obj['OtherFields'].get('defaultunit_povrsina_placa_d'))
        else:
            item['povrsina_placa'] = 0

        if data_obj['OtherFields'].get('grejanje_s'):
            item['grejanje'] = data_obj['OtherFields'].get('grejanje_s')
        else:
            item['grejanje'] = ""

        if data_obj['OtherFields'].get('grad_s'):
            item['grad'] = data_obj['OtherFields'].get('grad_s')
        else:
            item['grad'] = ""

        if data_obj['OtherFields'].get('lokacija_s'):
            item['lokacija'] = data_obj['OtherFields'].get('lokacija_s')
        else:
            item['lokacija'] = ""

        if data_obj['OtherFields'].get('mikrolokacija_s'):
            item['mikrolokacija'] = data_obj['OtherFields'].get('mikrolokacija_s')
        else:
            item['mikrolokacija'] = ""

        if data_obj['OtherFields'].get('defaultunit_kvadratura_d'):
            item['kvadratura'] = int(data_obj['OtherFields'].get('defaultunit_kvadratura_d'))
        else:
            item['kvadratura'] = -1

        if data_obj['OtherFields'].get('ostalo_ss') and "Parking" in data_obj['OtherFields'].get('ostalo_ss'):
            item['parking'] = "DA"
        else:
            item['parking'] = ""

        if data_obj['OtherFields'].get('dodatno_ss') and "Uknjižen" in data_obj['OtherFields'].get('dodatno_ss'):
            item['uknjizenost'] = "DA"
        else:
            item['uknjizenost'] = ""

        if data_obj['OtherFields'].get('ostalo_ss') and "Terasa" in data_obj['OtherFields'].get('ostalo_ss'):
            item['terasa'] = "DA"
        else:
            item['terasa'] = ""

        if data_obj['OtherFields'].get('ostalo_ss') and "Lift" in data_obj['OtherFields'].get('ostalo_ss'):
            item['lift'] = "DA"
        else:
            item['lift'] = ""

        item['tip_objekta'] = data_obj['OtherFields'].get('tip_objekta_s') or ""

        if data_obj['OtherFields'].get('sprat_s'):
            if "PR" in data_obj['OtherFields'].get('sprat_s'):
                item['sprat'] = 0
            else:
                item['sprat'] = data_obj['OtherFields'].get('sprat_s')
        else:
            item['sprat'] = 0

        if data_obj['OtherFields']['defaultunit_cena_d']:
            item['cena'] = int(data_obj['OtherFields']['defaultunit_cena_d'])
        else:
            item['cena'] = 0

        if data_obj['GeoLocationRPT'].split(",")[0]:
            item['x_pos'] = float(data_obj['GeoLocationRPT'].split(",")[0])
        else:
            item['x_pos'] = 0

        if data_obj['GeoLocationRPT'].split(",")[1]:
            item['y_pos'] = float(data_obj['GeoLocationRPT'].split(",")[1])
        else:
            item['y_pos'] = 0

        return item
