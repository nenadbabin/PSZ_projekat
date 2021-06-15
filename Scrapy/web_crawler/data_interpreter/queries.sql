-- select count(*) from psz_projekat.nekretnina where tip_ponude = 'Prodaja';
-- select count(*) from psz_projekat.nekretnina where tip_ponude = 'Izdavanje';

-- select * from psz_projekat.nekretnina where tip_ponude = 'Prodaja';
-- select * from psz_projekat.nekretnina where tip_ponude = 'Izdavanje';

-- select grad, count(*) from psz_projekat.nekretnina 
-- 					  where tip_ponude = 'Prodaja'
--                       group by grad
--                       order by count(*) desc;

-- select count(*) from psz_projekat.nekretnina where uknjizenost = 'DA' and tip_nekretnine = 'Stan';
-- select count(*) from psz_projekat.nekretnina where uknjizenost = '' and tip_nekretnine = 'Stan';
-- select count(*) from psz_projekat.nekretnina where uknjizenost = 'DA' and tip_nekretnine = 'Kuća';
-- select count(*) from psz_projekat.nekretnina where uknjizenost = '' and tip_nekretnine = 'Kuća';

-- select * from psz_projekat.nekretnina 
-- 		 where tip_nekretnine = 'Kuća' and tip_ponude = 'Prodaja'
--          order by cena desc
--          limit 30;
-- select * from psz_projekat.nekretnina 
-- 		 where tip_nekretnine = 'Stan' and tip_ponude = 'Prodaja'
--          order by cena desc
--          limit 30;

-- select * from psz_projekat.nekretnina 
-- 		 where tip_nekretnine = 'Kuća'
--          order by kvadratura desc
--          limit 100;
-- select * from psz_projekat.nekretnina 
-- 		 where tip_nekretnine = 'Stan'
--          order by kvadratura desc
--          limit 100;

-- select * from psz_projekat.nekretnina 
-- 		 where tip_objekta = 'Novogradnja'
--          order by cena desc

-- select * from psz_projekat.nekretnina
-- 		 order by broj_soba desc
--          limit 30;
-- select * from psz_projekat.nekretnina
-- 		 where tip_nekretnine = 'Kuća'
-- 		 order by povrsina_placa desc
--          limit 30;
-- select * from psz_projekat.nekretnina
-- 		 where tip_nekretnine = 'Stan'
-- 		 order by kvadratura desc
--          limit 30;

-- select lokacija, count(*) from psz_projekat.nekretnina
-- 				   where grad = 'Beograd'
--                    group by lokacija
--                    order by count(*) desc
--                    limit 10;


-- select * from psz_projekat.nekretnina
-- where tip_ponude = 'Prodaja' and grad = 'Beograd'nekretnina

select * from psz_projekat.nekretnina
where grad = 'Beograd' and tip_ponude = 'Prodaja' and tip_nekretnine = 'Stan';