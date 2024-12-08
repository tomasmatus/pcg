# PCG projekt 2
- autor: xmatus37

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] |
|:-----:|----------|------------|------------|
|  4096 | 0.492139 |   0.116796 |   0.078090 |
|  8192 | 1.471328 |   0.228242 |   0.152765 |
| 12288 | 2.478942 |   0.339821 |   0.227431 |
| 16384 | 3.386801 |   0.469618 |   0.336612 |
| 20480 | 5.059240 |   0.585854 |   0.420504 |
| 24576 | 7.112179 |   0.702756 |   0.504536 |
| 28672 | 9.892856 |   0.830380 |   0.589687 |
| 32768 | 12.59829 |   0.973591 |   0.687970 |
| 36864 | 15.54297 |   1.095055 |   0.774065 |
| 40960 | 19.36099 |   1.218446 |   0.860117 |
| 45056 | 23.48723 |   1.493099 |   1.113557 |
| 49152 | 27.69359 |   1.630885 |   1.222914 |
| 53248 | 32.63063 |   1.767688 |   1.326357 |
| 57344 | 37.43660 |   2.576564 |   1.823161 |
| 61440 | 42.85863 |   2.764267 |   1.952844 |
| 65536 | 49.46104 |   2.949443 |   2.083438 |
| 69632 | 55.14939 |   3.498231 |   2.573292 |
| 73728 | 62.04446 |   3.764923 |   2.762472 |
| 77824 | 69.26138 |   3.981206 |   2.917486 |
| 81920 | 76.60071 |   4.189266 |   3.073185 |

### Závěrečné
|    N   |  CPU [s] | GPU [s]  | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:--------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.065481 |  16.68881 |             0.13191 |        108.556 |
|   2048 |   0.5958 | 0.083530 |   7.13199 |             0.13239 |        220.773 |
|   4096 |   0.6652 | 0.119541 |   5.56462 |             0.13152 |        440.096 |
|   8192 |   1.6599 | 0.191354 |   8.67449 |             0.13046 |        876.449 |
|  16384 |   3.3655 | 0.364680 |   9.22863 |             0.11764 |      1 577.492 |
|  32768 |  12.7233 | 0.695351 |  18.29766 |             0.11420 |      3 072.485 |
|  65536 |  48.9732 | 1.974088 |  24.80801 |             0.07527 |      4 051.300 |
| 131072 | 195.9965 | 8.029187 |  24.03686 |             0.03926 |      4 153.638 |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**

Ano vyskytla, u `N = 57344` je vidět skok na dvojnásobný čas oproti předchozímu běhu.
Pro nízká `N` v rozmezí `4096 <= N < 57344` se data vejdou na všechny dostupné
SM procesory a výpočet se provede najednou.
U `N >= 57344` už grid obsahuje příliš mnoho blocků, takže se všechny nemohou
zároveň poslat na SM procesory a musí se čekat než první várka blocků dokončí výpočet
a pak se pošlou zbývající blocky na SM procesory.

### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**

Ano, došlo k výraznému zrychlení výpočtu.

**Popište hlavní důvody:**

Místo volání tři kernelů se volá pouze jeden, takže program má nižší overhead
spojený s voláním kernelů. Sloučení kernelů u tohoto algoritmu také přináší
lepší znovupoužítí dat a některé hodnoty se nemusí počítat opakovaně.
Je zde i lepší lokalita dat a podstatně menší počet přístupů do hlavní paměti,
protože nedojde k vyprázdnění registrů mezi voláními jednotlivých kernelů
a není tak nutné třikrát přistupovat do hlavní paměti pro získání hodnot
struktury `Particles`.

### Krok 2: Výpočet těžiště
**Kolik kernelů je nutné použít k výpočtu?**

Volají se dva různé kernely.
První kernel inicializuje buffer `comBuffer` vstupními hodnotami částic,
druhý kernel pak provádí paralelně redukci dokud `stride / 2 != 0`

Celkem se tedy spustí `log2(N) + 1` kernelů.

**Kolik další paměti jste museli naalokovat?**

Alokuje se nejbližší vyšší sudý počet N. Tedy alokuje se pole float4 o stejném počtu
jako je počet částic, případně N+1 pokud je počet částic lichý.
Sudá velikost pole je nutná, protože následně funkce pro výpočet těžíště iteruje
přes `velikosti pole / 2` a je tedy nutné, aby velikost byla celočístelně dělitelná dvojkou.

Nejvíce se tedy alokuje `(N + 1) * sizeof(float4)` bytů.

**Jaké je zrychelní vůči sekveční verzi? Zdůvodněte.** *(Provedu to smyčkou #pragma acc parallel loop seq)*

Dochází ke znáčnému zrychlení. Výpočet těžiště pro `N = 81920` trvá sekvenčnímu algoritmu na GPU
15ms a paralelizované verzi trvá stejný výpočet 0.2ms.
Důvodem zrychlení je, že v jednotlivých iterací může pracovat více vláken zároveň.
Ovšem počet vláken vždy odpovídá polovíně vláken právě zpracovávané velikosti pole,
takže počet vláken postupně ubívá až v poslední iteraci pracuje pouze jedno vlákno,
které provede poslední výpočet.

### Krok 4: Měření výkonu
**Jakých jste dosáhli výsledků?**

Dosáhl jsem pěkného zrychlení, dokonce většího než má předchozí implementace v CUDA.
Upřímně z takového výsledku nejsem velice překvapený, asi bych očekával, že kombinace
OpenACC a síly překladače bude mnohem lepší než nováček v CUDA.

Využítí paměti a maximální GFLOPS celkem stagnují u posledních dvou největších velikostí
vstupních dat. Domnívám se, že to způsobuje nepoužítí sdílené paměti jejíž využití je v
OpenACC slabé oproti cílené implementaci v CUDA. Nicméně stále jde o velmi dobré výsledky
oproti CPU implementaci.

Propustnost paměti je oprotí dřívější implementaci v CUDA také větší.
Tady se také domnívám, že je to způsobeno menším použití sdílené paměti.

Pozn.:

Během testování posledního kroku jsem si všiml, že nekolik málo mezivýsledků
průběžných zápisů nesplní požadovanou přesnost výpočtu a skript `compare.sh`
je označí jako nesrovnalosti vůči CPU výstupu. Vždy se jedná o opravdu minimální
odchylky a předpokládám, že jde o kumulativní chybu operací nad floaty.
Bohužel se mi výpočet nepodařilo upravit tak, abych těchto pár nesrovnalostí odstranil.

**Lze v datech pozorovat nějaké anomálie?**

Případá mi zvláštní, že výpočet na GPU pro `N=1024` dosahuje tak vysokého zrychlení.
Očekával bych, že kvůli potřebě kopírovat data na GPU, spouštět kernel a kopírovat
data zpět z GPU na CPU vnese dostatečně vysoký overhead kvůli kterému by nebylo
výhodné takto malý výpočet provádět na GPU.