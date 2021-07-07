# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface

import mysql.connector


class WebCrawlerPipeline:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="nenad",
            database="psz_projekat"
        )

        self.cursor = self.conn.cursor()

    def process_item(self, item, spider):

        sql_query = "replace into nekretnina (" \
                    "tip_ponude," \
                    "tip_nekretnine," \
                    "broj_soba," \
                    "spratnost," \
                    "sprat," \
                    "povrsina_placa," \
                    "grejanje," \
                    "grad," \
                    "lokacija," \
                    "mikrolokacija," \
                    "kvadratura, " \
                    "parking, " \
                    "uknjizenost," \
                    "terasa," \
                    "lift," \
                    "tip_objekta," \
                    "cena," \
                    "x_pos," \
                    "y_pos) " \
                    "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

        values = (item['tip_ponude'],
                  item['tip_nekretnine'],
                  item['broj_soba'],
                  item['spratnost'],
                  item['sprat'],
                  item['povrsina_placa'],
                  item['grejanje'],
                  item['grad'],
                  item['lokacija'],
                  item['mikrolokacija'],
                  item['kvadratura'],
                  item['parking'],
                  item['uknjizenost'],
                  item['terasa'],
                  item['lift'],
                  item['tip_objekta'],
                  item['cena'],
                  item['x_pos'],
                  item['y_pos']
                  )

        self.cursor.execute(sql_query, values)
        self.conn.commit()

        return item
