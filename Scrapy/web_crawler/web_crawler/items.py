# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class WebCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    naslov = scrapy.Field()
    tip_ponude = scrapy.Field()
    tip_nekretnine = scrapy.Field()
    broj_soba = scrapy.Field()
    spratnost = scrapy.Field()
    sprat = scrapy.Field()
    povrsina_placa = scrapy.Field()
    grejanje = scrapy.Field()
    grad = scrapy.Field()
    lokacija = scrapy.Field()
    mikrolokacija = scrapy.Field()
    kvadratura = scrapy.Field()
    parking = scrapy.Field()
    uknjizenost = scrapy.Field()
    terasa = scrapy.Field()
    lift = scrapy.Field()
    tip_objekta = scrapy.Field()
    cena = scrapy.Field()
