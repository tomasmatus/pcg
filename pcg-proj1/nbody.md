# PCG projekt 1
- autor: xmatus37

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|----------|------------|------------|------------|
|  4096 | 0.492139 |   0.295600 |   0.185137 |   0.156735 |
|  8192 | 1.471328 |   0.592127 |   0.371814 |   0.312290 |
| 12288 | 2.478942 |   0.888212 |   0.557963 |   0.467899 |
| 16384 | 3.386801 |   1.184411 |   0.743707 |   0.623454 |
| 20480 | 5.059240 |   1.479231 |   0.929744 |   0.779177 |
| 24576 | 7.112179 |   1.775256 |   1.115525 |   0.934717 |
| 28672 | 9.892856 |   2.072021 |   1.302106 |   1.090502 |
| 32768 | 12.59829 |   2.367432 |   1.488100 |   1.246531 |
| 36864 | 15.54297 |   2.664146 |   1.674138 |   1.402208 |
| 40960 | 19.36099 |   2.960279 |   1.860717 |   1.557778 |
| 45056 | 23.48723 |   3.256293 |   2.046458 |   1.713697 |
| 49152 | 27.69359 |   3.551571 |   2.232185 |   1.869465 |
| 53248 | 32.63063 |   3.848073 |   2.418717 |   2.024823 |
| 57344 | 37.43660 |   6.618256 |   4.321085 |   3.669833 |
| 61440 | 42.85863 |   7.090897 |   4.638775 |   3.932273 |
| 65536 | 49.46104 |   7.573050 |   4.956376 |   4.196333 |
| 69632 | 55.14939 |   8.046098 |   5.265466 |   4.463156 |
| 73728 | 62.04446 |   8.519320 |   5.575367 |   4.726500 |
| 77824 | 69.26138 |   8.997015 |   5.890790 |   4.989407 |
| 81920 | 76.60071 |   9.467184 |   6.202047 |   5.250032 |

### Závěrečné
|    N   |  CPU [s] |   GPU [s] | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:---------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 |  0.041675 |     26.22 |             0.08114 |        138.224 |
|   2048 |   0.5958 |  0.080734 |      7.38 |             0.07197 |        280.090 |
|   4096 |   0.6652 |  0.158779 |      4.19 |             0.06873 |        564.184 |
|   8192 |   1.6599 |  0.314854 |      5.27 |             0.06567 |      1 131.507 |
|  16384 |   3.3655 |  0.627044 |      5.37 |             0.06415 |      2 267.253 |
|  32768 |  12.7233 |  1.251826 |     10.16 |             0.06339 |      4 538.837 |
|  65536 |  48.9732 |  4.209958 |     11.63 |             0.03743 |      5 387.500 |
| 131072 | 195.9965 | 12.539515 |     15.63 |             0.02499 |      7 221.229 |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**

Ano vyskytla, u `N = 57344` je vidět skok na dvojnásobný čas oproti předchozímu běhu.
Pro nízká `N` v rozmezí `4096 <= N < 57344` je v gridu dostatečně malý počet blocků
s vlákny, takže nedojde k úplnému zaplnění všech SM procesorů a výpočet se provede
najednou. U `N >= 57344` už grid obsahuje příliš mnoho blocků, takže se všechny nemohou
zároveň poslat na SM procesory a musí se čekat než první várka blocků dokončí výpočet
a pak se pošlou zbývající blocky na SM procesory.


### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**

Ano, došlo k výraznému zrychlení výpočtu.

**Popište hlavní důvody:**

Volá se pouze jeden kernel, takže nižší overhead, který přináší volání více kernelů.
Lepší práce s regisrty, protože se stejné hodnoty využívají v rámci jednoho vlákna
počítající v jednom kernelu narozdíl od rozdělení do tří kernelů, kde se hodnoty registrů
vyprázdní. Také došlo k redukování počtu volání funkce sqrt, která je pro výpočet na GPU
náročná.

### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**

Ano došlo k celkem znatelnému zrychlení, ale ne tak výraznému jako během sloučení kernelů.
Zrychlení je více znatelné u většího počtu vstupních dat.

**Popište hlavní důvody:**

K zrychlení došlo především díky znovupoužítí již načtené hodnoty. Díky tomu ubylo
load operací z globální paměti a SM procesory čekají na data kratší dobu. 

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**

Dosáhl jsem celkem slušného zrychlení v porovnání s referenční implementací.
Nicméně dle profilingu je využití SM procesorů na 60% v případě vstupu
o velikosti `N=131072`, takže jsem ještě nedokázal vytěžit maximum z GPU
a rychlost by se dala vylepšit.

S větší velikostí vstupu od velikosti `N=32468` dosahuje GPU vyššího
zrychlení a s růstem počtu vstupních dat se toto zrychlení zlepšuje
v porovnání s CPU.

**Lze v datech pozorovat nějaké anomálie?**

Případá mi zvláštní, že výpočet na GPU pro `N=1024` dosahuje zrychlení 26.22x.
Očekával bych, že kvůli potřebě kopírovat data na GPU, spouštět kernel a kopírovat
data zpět z GPU na CPU vnese dostatečně vysoký overhead kvůli kterému by nebylo
výhodné takto malý výpočet provádět na GPU.
