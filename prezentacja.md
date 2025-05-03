# Wprowadzenie

Język flagowy Mojo jest flagowym produktem firmy Modular, zjamującej się tworzeniem narzędzi i technologii dla programistów sztucznej inteligencji. Jego celem jest połączenie wydajności C++ z prostotą i ekosystemem Pythona. Dzięki umożliwieniu programowania zarówno pod CPU oraz akceleratory GPU, znajduje on zastosowanie w uczeniu maszynowym i obliczeniach numerycznych.
![Logo Mojo](https://cdn.prod.website-files.com/63f9f100025c058594957cca/65df9332b319cb9989698874_mojo.jpg)
![Logo Modular](https://cdn.prod.website-files.com/64174a9fd03969ab5b930a08/642853d18fbcbe9695c3f343_64078db03b0c895dbc658746_mod-word-mark.png)
![Logo Python](https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png)

## Geneza

Twórcą firmy Modular oraz języka Mojo jest Chris Lattner – **_programista odpowiedzialny za stworzenie toolchainu LLVM oraz języka programowania Swift_**(Do zmiany nie podoba mi się to zdanie). Jego prace miały ogromny wpływ na świat kompilatorów i nowoczesnych języków programowania. Mojo to kolejny krok w tej ewolucji – język stworzony specjalnie do wydajnego programowania niskopoziomowego, stricte pod kątem akceleratorów AI takich jak GPU, TPU, NPU itp.

![YT](https://i.ytimg.com/vi/-8TbsCUuwQQ/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCoYhk-zm8iz3C7aiqJJU8NidosJQ)

<center style={{ display: 'flex' }}>

![LLVM](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZsvm8rP2Wlxqe3U2uTTbJ5X4MkXeMF57N9w&s)

![Swift](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRd4u7UgyLvk9FitFFZsUjP0o__BHUlDzmmeQ&s)

</center>

## GPU TPU NPU porównanie

Mojo został zaprojektowany jako hybrydowy język programowania, który łączy przystępność składni Pythona z wydajnością niskopoziomowych języków jak C++. Jego głównym założeniem jest efektywne wykorzystanie specjalistycznych akceleratorów obliczeniowych (GPU, TPU, NPU) poprzez automatyczną optymalizację na poziomie kompilatora z wykorzystaniem MLIR. W przeciwieństwie do tradycyjnych rozwiązań niskopoziomowych, Mojo eliminuje potrzebę ręcznego zarządzania wieloma aspektami sprzętowymi, zachowując przy tym pełną kontrolę nad wydajnością i zarządzaniem pamięcią.

![Logo OpenCl](https://miro.medium.com/v2/resize:fit:910/1*eaiku5FlPeoeStXppsTQzg.png)

```cpp
// Kod OpenCL przypomina C/C++
__kernel void vector_add(__global const float *a,
                       __global const float *b,
                       __global float *c,
                       const unsigned int n) {
  int id = get_global_id(0);
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
```

![Logo Nividia Cuda](https://static.wikia.nocookie.net/logopedia/images/1/1f/Nvidia_CUDA.svg/revision/latest?cb=20230319014140)

```cpp
// Kod CUDA przypomina C/C++
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
```

```py
# Kod Mojo przypomina Python
fn main():
    var v1 = SIMD[DType.uint8, 4](1, 2, 3, 4)
    var v2 = SIMD[DType.uint8, 4](4, 3, 2, 1)
    var v3 = v1 + v2
    print(v3)
```

Jedną z kluczowych zalet Mojo jest jego kompilator, który automatycznie optymalizuje kod pod sprzęt docelowy (target device). Aby to osiągnąć, Mojo korzysta z technologii opracowanej przez Google w ramach LLVM, zwanej Multi-Level Intermediate Representation (MLIR).

<center>

![Logo MLIR](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSZ1c3YQ7AN8NlrdiTf-xCdcGnZxovlqKQ6QA&s)

</center>

## Czym jest MLIR?

MLIR to rozwiązanie opracowane przez Google w ramach LLVM, które sprowadza kod do wspólnej reprezentacji dla wielu poziomów abstrakcji. Dzięki temu umożliwia:

- Konwersję operacji tensorowych i matematycznych na najbardziej efektywne wersje,

- Automatyczne dostosowanie kodu do konkretnego sprzętu (scalanie operacji tensorowych dla TPU, czy lepsza paralelizacja dla GPU).

MLIR pozwala więc na łatwe i wydajne programowanie akceleratorów, eliminując potrzebę ręcznego dostrajania kodu pod konkretne urządzenia.

## Konkretniej jak działa

MLIR (Multi-Level Intermediate Representation) działa poprzez sprowadzenie operacji do symbolicznej reprezentacji, co pozwala na analizę i transformację kodu na różnych poziomach abstrakcji. Przykładowo, prosty program w Pythonie, taki jak print(42), może zostać sprowadzony do ciągu instrukcji maszyny wirtualnej, które są bardziej szczegółowe i bliższe niskiemu poziomowi wykonania.

```
# Załaduj stałą 42 do rejestru R1
OP_LOAD_CONST R1, 42
# Załaduj funkcję 'print' z globalnego zakresu do rejestru R2
OP_LOAD_GLOBAL R2, "print"
# Wywołaj funkcję 'print' z argumentem w R1
OP_CALL R2, [R1]
# Zwróć None (wartość domyślna w Pythonie)
OP_RETURN None
# Zatrzymaj wykonanie
OP_STOP
```

Ten kod, reprezentujący operacje na maszynie wirtualnej, może zostać dalej przetworzony na reprezentację MLIR, która jest bardziej abstrakcyjna i umożliwia głębszą analizę. Operacje te są na tyle fundamentalne, że można je sprowadzić do reprezentacji MLIR, co ułatwia dalszą analizę i transformację kodu.

```
module @python_print_42 {
  func @main() -> !python.object {
    %const_42 = python.const 42 : !python.int  // Załaduj stałą 42
    %print_func = python.load_global "print" : !python.object // Załaduj funkcję 'print'
    %result = python.call %print_func(%const_42) : (!python.object, !python.object) -> !python.object // Wywołaj print(42)
    return %result : !python.object // Zwróć wynik (None)
  }
}
```

W MLIR, przestrzeganie pewnych zasad, takich jak Single Static Assignment (SSA), jest kluczowe dla efektywnej analizy i optymalizacji kodu. SSA gwarantuje, że każda zmienna jest przypisywana tylko raz, co ułatwia śledzenie zależności danych i wykonywanie transformacji. Dzięki temu, MLIR może wykonywać zaawansowane analizy, takie jak analiza przepływu danych, analiza zależności oraz optymalizacje, które poprawiają wydajność kodu.

## Dialekty

Dialekty to konstrukcje reprezentujące operacje specyficzne dla danej domeny. Przykładowym dialektem jest %print_func który reprezentuje "Coś co printuje". Poprzez uzupełnienie tych fundamentalnych bloków własnymi ich definicjami, przykładowo

```
printf // C
console.log // JS
puts // Ruby
Decir // Hiszpański
```

Co umożliwia przekompilowanie programu z jednego języka programowania na inny, bardziej odpowiadający problemom domenowym, które chcemy rozwiązać. Przykładowo, docelowym językiem może być Intermediate Representation (IR) dla LLVM (kod maszynowy) lub NVVM (CUDA). Dzięki tej sztuczce, Mojo może osiągnąć wydajność porównywalną z C++, pisząc program w języku podobnym do Pythona, co jest kluczowe dla wygodnego dostępu do jego bibliotek i abstrakcji.

Jest to zdecydowanie przyjemniejsze i wydajniejsze niż korzystanie z Numpy poprzez Serwer HTTP, FFI, czy inną warstwę pośredniczą.

## Control Flow Graph

Dialekty w MLIR oferują jeszcze jedną potężną możliwość: umożliwiają tworzenie grafów przepływu sterowania (CFG). CFG to reprezentacja grafowa, która pokazuje, jak sterowanie przepływa przez program. Wierzchołki grafu reprezentują bloki kodu, a krawędzie reprezentują możliwe przejścia między tymi blokami. Dzięki tej reprezentacji, MLIR jest w stanie wydedukować i analizować różne ścieżki wykonania programu, co otwiera drzwi do zaawansowanych optymalizacji.

![Graf Dialektów](https://lowlevelbits.org/img/compiling-ruby-3/what-is-mlir.png)

Istnieje wiele zaawansowanych technik optymalizacji, takich jak fuzja tensorów, która pozwala na łączenie wielu operacji tensorowych w jedną, bardziej złożoną operację. Takie podejście znacząco redukuje narzut związany z uruchamianiem wielu pojedynczych operacji, co jest szczególnie istotne w kontekście obliczeń na GPU. Reprezentacja pośrednia (IR), taka jak ta stosowana w MLIR, ułatwia kompilatorowi wykrycie możliwości fuzji tensorów, ponieważ operacje są reprezentowane w sposób symboliczny i strukturalny. To pozwala na analizę zależności danych i identyfikację sekwencji operacji, które można połączyć.

![Tensor Fusion](https://www.researchgate.net/publication/362952579/figure/fig3/AS:11431281256139673@1719510347593/The-encoding-module-based-on-tensor-fusion.jpg)

Dodatkowo, reprezentacja pośrednia umożliwia efektywną paralelizację obliczeń na GPU. Dzięki analizie grafu przepływu danych, kompilator może automatycznie generować kod, który wykorzystuje równoległe możliwości GPU. Operacje tensorowe, które są niezależne od siebie, mogą być wykonywane równocześnie na różnych rdzeniach GPU, co znacznie przyspiesza obliczenia. Fuzja tensorów dodatkowo zwiększa ten efekt, ponieważ łączenie operacji zmniejsza liczbę uruchomień jądra GPU, co minimalizuje narzut związany z synchronizacją i przesyłaniem danych.

![Paralelizacja](https://bstncdn.net/i/1498)

W rezultacie, kompilator, który operuje na odpowiedniej reprezentacji pośredniej, jest w stanie _automatycznie_ wykryć nasze intencje i zastosować zaawansowane optymalizacje, takie jak fuzja tensorów i paralelizacja GPU, co prowadzi do znacznego wzrostu wydajności. To jest szczególnie ważne w kontekście uczenia maszynowego i innych aplikacji, które wymagają intensywnych obliczeń tensorowych. Czyli takich do których stworzony został język Mojo.

## Ale dlaczego Python??

Mimo swoich licznych wad, niepodważalną zaletą pythona jest ogromny i dojrzały ekosystem. Biblioteki do ML/AI, takie jak TensorFlow, PyTorch, oraz do big data, takie jak Pandas, PySpark sprawiają, że stał się standardem w tych dziedzinach. Mojo stara się czerpać z tej siły, ale jednocześnie usuwa ograniczenia wydajnościowe Pythona, pozwalając na jego wykorzystanie w programowaniu niskopoziomowym.
Dodatkowo, jego minimalistyczna i zwięzła forma idealnie nadaje się do analizy i optymalizacji przez MLIR.

Przykładowe użycie paczek pythonowych w Mojo
```py
from python import Python

def main():
    var np = Python.import_module("numpy")
    var arr = np.array(Python.list(1, 2, 3))
    print(arr)
```

![logo Pytorch](https://miro.medium.com/v2/resize:fit:1200/1*r2eKvfvYPQuizKLOh9q7Hw.jpeg)

<center>

![Logo Pandas](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-4t7O82hSHWJNpROgVI3ae7dcrRfEHUNoRg&s)

![Logo Pip](https://miro.medium.com/v2/resize:fit:300/0*lmWLpGJfUo-OnpC2.jpg)

</center>

Mojo jest skierowane do programistów i naukowców, którzy potrzebują wydajności zbliżonej do C++, ale nie są gotowi rezygnować z znajomej prostoty i ekosystemu pythona. Są to między innymi

- Programiści AI/ML: Tworzenie wydajnych modeli uczenia maszynowego i aplikacji AI.
- Naukowców danych: Analiza dużych zbiorów danych z wykorzystaniem bibliotek Pythona, ale z większą wydajnością.
- Programiści systemów: Tworzenie niskopoziomowych aplikacji, które wymagają wysokiej wydajności.
- Ogół programistów pythona: Chcących pisać kod Pythona, który jest szybszy i bardziej wydajny.

Mojo ma na celu wypełnienie luki między łatwością użycia Pythona a wydajnością C++, co czyni go atrakcyjnym wyborem dla szerokiego grona programistów.

### Alternatywy

W kontekście Pythona istnieje już kilka alternatywnych podejść, które próbują rozwiązać problem wydajności, choć każde z nich robi to w nieco inny sposób. Oto kilka z nich i porównanie z Mojo.
Mojo, jako nowy język programowania, który łączy łatwość użycia Pythona z wydajnością C++, stawia sobie ambitny cel: wypełnienie luki między tymi dwoma światami. W kontekście Pythona istnieje już kilka alternatywnych podejść, które próbują rozwiązać problem wydajności, choć każde z nich robi to w nieco inny sposób. Oto kilka z nich i porównanie z Mojo:

1. Cython - język programowania, który również jest nadzbiorem Pythona. Pozwala na pisanie kodu, który jest kompilowany do C, co znacznie przyspiesza jego wykonanie. Jest szeroko stosowany do optymalizacji istniejących bibliotek Pythona, takich jak NumPy.

Porównanie z Mojo:
- Cython wymaga nauki dodatkowej, specyficznej składni i ręcznej optymalizacji. Efektywne go wykorzystanie jest czasochłonne i wymaga zaawansowanej wiedzy oraz lat doświadczenia

- Mojo oferuje głębszą integrację z MLIR, dzięki czemu jest w stanie osiągnąć lepszą wydajność. Wykorzystuje do tego analizę grafu przepływu by wprowadzić zaawansowane optymalizacje w miejscach, w których nawet najbardziej doświadczeni programiści zawiodą.

<center>

![Logo Cython](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT3k74gA2JM9rFe0C7_GrV2xB03X_jm_6Z87w&s)

</center>

2. Numba - kompilator just-in-time (JIT) dla Pythona, który tłumaczy kod Pythona na zoptymalizowany kod maszynowy w czasie wykonywania.
   Jest szczególnie skuteczny w optymalizacji obliczeń numerycznych i operacji na macierzach, co jest kluczowe w uczeniu maszynowym. Porównanie z Mojo:

- Numba działa w czasie wykonywania, co może powodować pewien narzut związany z kompilacją JIT. Mojo kompiluje kod statycznie, co pozwala na lepsze optymalizacje w czasie kompilacji i eliminuje narzut w czasie wykonywania.

![Logo Numba](https://inlocrobotics.com/wp-content/uploads/2021/06/numba-1.jpg)

3. PyPy - alternatywna implementacja Pythona, która wykorzystuje kompilator JIT do przyspieszenia wykonywania kodu. Jest szczególnie skuteczny w optymalizacji kodu, który wykonuje wiele operacji dynamicznych, takich jak operacje na listach i słownikach. Porównanie z Mojo:

- PyPy jest kompatybilny z istniejącym kodem Pythona, ale nie oferuje tak dużych możliwości optymalizacji niskopoziomowej jak Mojo, celującego w umożliwienie programowania niskopoziomowego z zachowaniem prostoty Pythona, co pozwala na pisanie kodu, który jest zarówno wydajny, jak i czytelny.

![Logo PyPy](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/Pypy-logo_%282020%29.svg/2560px-Pypy-logo_%282020%29.svg.png)

4. JAX - biblioteka Pythona do transformacji programów numerycznych.
   umożliwia automatyczną dyferencjację, wektoryzację i kompilację kodu Pythona do kodu przyspieszonego na GPU/TPU. Porównanie z Mojo:

- Mimo że JAX jest imponujący w swojej dziedzinie, jest ograniczony byciem jedynie biblioteką w ekosystemie Pythona. Oznacza to, że jego możliwości są ograniczone do transformacji programów numerycznych i nie oferuje pełnej kontroli nad niskopoziomowymi aspektami programowania, jak robi to Mojo. Efektywne użycie Mojo nie wymaga czasu spędzonego na optymalizacji i dogłębnej wiedzy o transformacjach numerycznych, osiągając przy tym nie gorsze rezultaty.

![Logo JAX](https://preview.redd.it/the-jax-logo-v0-mcrezb1fy9bb1.png?auto=webp&s=81b1cd55dd5b0580081bf2bb188dc65849a101c7)

# W praktyce

Mojo jest aktualnie dostępny na systemach Linux oraz MacOS. Eksperymentalne wsparcie systemu Windows oparte jest na zintegrowanym środowisku Windows Subsystem for Linux (Nam nie udało się na nim wiele zdziałać). Skorzystać z Mojo możemy poprzez narzędzie CLI Magic, które służy do interakcji z produktami firmy Modular. Jego instalacja sprowadza się do wykonania zapytania CURL pod odpowiedni adres (Jest on generowany automatycznie po wejściu w dokumentację)

TODO:
#Linux logo#
#MacOS logo#
#WSL logo tylko przekreślone bo to był bolesny żart#

```bash
curl -ssL https://magic.modular.com/deb13af9-76a7-4aa4-b9a3-98fc64f58c8e | bash
```

## Hello World

Aby utworzyć najprostszy program, potrzebujemy zainicjować środowisko wirtualne, działające analogicznie do tych z ekosystemu pythona

```bash
magic init #NAZWA PROJEKTU# --format mojoproject
```

Po zainicjalizowaniu środowiska, magic dodaje nam paczke 'max' jako zależność, która zawiera mojo.
Teraz musimy uruchomić wirtualne środowisko w obecnym katalogu

```bash
magic shell
```

Do uruchomienia programu użyjemy jednej z dwóch komend:

- mojo run <nazwa_pliku> -- Uruchamia wskazany program.
- mojo build <nazwa_pliku> -- Tworzy plik wykonywalny z projektu.

Kod wykonywany znajdzie się w funkcji o nazwie main (analogicznie w C++)

```mojo
fn main():
  print("Hello, world")
```

## System typów w Mojo

Jednym z filarów Mojo jest jego elastyczny system typów, który pozwala na balansowanie między wygodą dynamicznego typowania (jak w Pythonie), a bezpieczeństwem i wydajnością statycznego typowania (jak w C++).

Mojo umożliwia używanie:

- Typów statycznych (`Int`, `Float64`, `Bool`, `Struct`) — wykorzystywanych do maksymalizacji wydajności
- Typów dynamicznych — przez pełną interoperacyjność z Pythonem
- Typów generowanych w czasie kompilacji (`let`, `var`, `const`) — umożliwiają precyzyjne kontrolowanie mutowalności

```mojo
fn add(a: Int, b: Int) -> Int:
    return a + b

```

## Przykład 1: Fibbonacci

```py
from python import Python
from tensor import Tensor

# Mnożenie macierzy 2x2
fn mat_mult(A: Tensor[DType.int64], B: Tensor[DType.int64]) -> Tensor[DType.int64]:
    var t = Tensor[DType.int64]((2,2))
    t.store(0, A[0,0] * B[0,0] + A[0,1] * B[1,0])
    t.store(1, A[0,0] * B[0,1] + A[0,1] * B[1,1])
    t.store(2, A[1,0] * B[0,0] + A[1,1] * B[1,0])
    t.store(3, A[1,0] * B[0,1] + A[1,1] * B[1,1])
    return t

# Potęgowanie macierzy
fn mat_pow(Q: Tensor[DType.int64], n: Int) -> Tensor[DType.int64]:
    var result = Tensor[DType.int64]((2, 2))
    result.store(0, 1)
    result.store(3, 1)
    var base = Q

    var exp = n
    while exp > 0:
        if exp % 2 == 1:
            result = mat_mult(result, base)
        base = mat_mult(base, base)
        exp = exp // 2

    return result

# Tworzenie macierzy Q
def get_fib_mat() -> Tensor[DType.int64]:
  var t = Tensor[DType.int64]((2, 2))
  t.store(0, 1)
  t.store(1, 1)
  t.store(2, 1)

  return t


def main():
  a = get_fib_mat()
  x = Python.evaluate("[]")
  y = Python.evaluate("[]")
  outer_n = -1
  for n in range(1, 200):
    v = mat_pow(a, n)[1].__int__()
    if (v < 0):
      # Integer overflow
      outer_n = n
      break
    x.append(n)
    y.append(v)
    print(n)

  print(outer_n, "liczba fibonacciego powoduje integer overflow")
```

Widoczne różnice:

- Deklarowane zmienne muszą posiadać typy.
- Operujemy najczęściej na mało wygodnych Tensorach
- Mojo ma własny zestaw bibliotek, zależności pythona możemy importować korzystając z modułu Python.

Warto też dodać, że Tensor w Mojo został napisany w taki sposób, aby wykorzystywał instrukcje SIMD, tam gdzie jest to możliwe, co pozwala na równoległe przetwarzanie danych na macierzach. Mojo jest też w stanie wywnioskować rozmiar rejestrów SIMD na podstawie architektury obliczeniowej użytkownika.


## Porównanie z CUDA

CUDA wymaga znajomości C/C++ oraz specyficznych konstrukcji NVIDIA, co tworzy wysoką barierę wejścia dla specjalistów data science, którzy zazwyczaj mają doświadczenie tylko w Pythonie. Dodatkowo działa tylko na kartach NVIDIA.

Mojo oferuje składnie podobną do Pythona, zachowując przy tym możliwość mikrozarządzania pamięcią i wydajnością. Dzięki MLIR może kierować kod na różne akceleratory (GPU, TPU, NPU) bez zmian w kodzie źródłowym.

## Przykład - Dodawanie elementów macierzy

```c
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

int main() {
    const int size = 5;
    int a[size] = {1, 2, 3, 4, 5};
    int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {0};

    // Zaalokuj pamięć na GPU
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, size * sizeof(int));
    cudaMalloc(&dev_b, size * sizeof(int));
    cudaMalloc(&dev_c, size * sizeof(int));

    // Przekopiuj dane na GPU
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Uruchom kernel 'add'
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b, size);

    // Przekopiuj wynik z powrotem na CPU
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    // Zwolnij pamięć
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}
```

- \_\_global\_\_ - specyfikator oznaczający, że funkcja jest kernelem wykonywanym na GPU
- blockIdx, blockDim, threadIdx - zmienne wbudowane określające pozycję wątku w hierarchii GPU
- cudaMalloc - alokuje pamięć na GPU. Przyjmuje adres wskaźnika (stąd &dev_a) i rozmiar do zaalokowania
- cudaMemcpy - kopiuje dane między pamięcią CPU i GPU
- <<<numBlocks, threadsPerBlock>>> - składnia uruchamiania kernela z konfiguracją wątków
- cudaFree - zwalnia pamięć zaalokowaną na GPU (podobnie jak free() dla standardowej pamięci CPU).

```py
from memory import UnsafePointer
from algorithm import parallelize

alias size = 5
alias type = DType.int32

fn main():
    # Dynamiczne listy w Mojo
    var a = UnsafePointer[Scalar[type]].alloc(size)
    var b = UnsafePointer[Scalar[type]].alloc(size)
    var c = UnsafePointer[Scalar[type]].alloc(size)

    # Równoległe inicjowanie tablic (symulując CUDA kernel)
    @parameter
    fn init_arrays(i: Int):
        a.store(i, i + 1)
        b.store(i, (i + 1) * 10)
        c.store(i, 0)
    parallelize[init_arrays](size, size)

    # Równoległe dodawanie elementów tablic (symulując CUDA kernel)
    @parameter
    fn add_vectors(i: Int):
        c.store(i, a.load(i) + b.load(i))

    # Równoległe dodawanie elementów tablic
    parallelize[add_vectors](size, size)

    for i in range(size):
        print(a.load(i), "+", b.load(i), "=", c.load(i))

    a.free(); b.free(); c.free();
```

- UnsafePointer - pozwala na bezpośrednie zarządzanie pamięcią (podobnie jak wskaźniki w C++)
- parallelize - funkcja, która umożliwia równoległe wykonywanie kodu (podobnie jak CUDA)
- @parameter - dekorator funkcji, który oznacza, że funkcja może być przekazana jako parametr funkcji wyższego rzędu

## Benchmark: Mojo vs Python

Mojo celuje w wydajność zbliżoną do C. Przykład prostego benchmarku pokazuje różnicę między Pythonem a Mojo dla funkcji rekurencyjnej:

Python:
```py
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

def main():
    n = 30
    _ = fib(n)
```

```bash
hyperfine -N --warmup 100 --runs 100 'python3 main.py'

Benchmark 1: python3 main.py
  Time (mean ± σ):     260.2 ms ±  25.4 ms    [User: 255.5 ms, System: 4.3 ms]
  Range (min … max):   224.3 ms … 344.5 ms    100 runs
```

Mojo:
```mojo
fn fib(n: Int) -> Int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main():
    var n = 30
    _ = fib(n)
```

```bash
hyperfine -N --warmup 200 --runs 100 ./main

Benchmark 1: ./main
  Time (mean ± σ):      21.3 ms ±   0.7 ms    [User: 13.1 ms, System: 5.5 ms]
  Range (min … max):    20.1 ms …  24.1 ms    100 runs
```

Jak widać, w takim prostym przypadku, gdzie kod jest niemal identyczny i nie wykorzystaliśmy żadnych optymalizacyjnych sztuczek, które Mojo oferuje, to Mojo jest dużo szybsze od Pythona. Jest to głównie zasługa kompilacji Mojo do niskopoziomowego kodu, którego czas wykonania jest znacznie krótszy niż w przypadku Pythona, który musi interpretować kod w czasie wykonywania. Jak i również zastosowania wielu metod optymalizacji, m.in uniknięcie sprawdzania typów, dzięki statycznemu typowaniu, tail call optimization dla rekurencji i branch prediction.

## TODO: wiecej przykładów i bardziej szczegółowe przejscie przez jakis fajny przykład, aby kupic czas i pokazanie benchmarków

## TODO: moze jakis przyklad z modelem ML?

## TODO: przykład jak działa computation graph, lazy-execution, parallel

## TODO: benchmark z GPU, ale wynając maszyne z GPU, które wspiera mikroarchitekture Ampere

## TODO: kernele

https://docs.modular.com/max/tutorials/build-custom-ops

https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python/

## Automatyczna optymalizacja

- CUDA wymaga ręcznego zarządzania pamięcią, planowania wątków i optymalizacji - programista musi dokładnie rozumieć architekturę GPU
- Mojo dzięki MLIR automatycznie optymalizuje kod pod docelowy akcelerator, wykonując fuzję tensorów i paralelizację

## Doświadczenie

Wygląd może mylić - to nie jest Python. Każda operacja wymaga przemyślenia i świadomości co się dzieje na niskim poziomie. Choć składnia Mojo jest podobna do Pythona, programowanie w tym języku wymaga zupełnie innego podejścia. Programista musi być świadomy:

- Zarządzania pamięcią - kiedy alokować i zwalniać zasoby
- Typów danych - statyczne typowanie jest w Mojo kluczowe dla wydajności
- Przepływu danych - optymalne wykorzystanie GPU/TPU wymaga świadomego projektowania przetwarzania danych
- Równoległości - wiele operacji może być wykonywanych jednocześnie, co wymaga odpowiedniego planowania

Społeczność Mojo jest wciąż w fazie rozwoju. Dokumentacja jest nierówna - niektóre części są szczegółowo opisane, inne pozostają wybiórczo opracowane lub wręcz pominięte. Ponieważ język jest w fazie intensywnego rozwoju, trudno znaleźć przykłady które działają bez konieczności wprowadzania licznych modyfikacji. Większość materiałów edukacyjnych pochodzi bezpośrednio od firmy Modular, a liczba tutoriali stworzonych przez społeczność jest nadal ograniczona.

Platformy do wymiany wiedzy, takie jak StackOverflow czy GitHub, mają niewiele pytań i przykładów kodu dla Mojo w porównaniu z dojrzałymi językami. Większość problemów trzeba rozwiązywać samodzielnie, eksperymentując z kodem i analizując dostępną dokumentację.

## Zastosowania

Mojo znajduje zastosowanie przede wszystkim w obszarach wymagających intensywnych obliczeń i wysokiej wydajności:
Uczenie maszynowe

### Trenowanie modeli neural networks z wykorzystaniem akceleratorów GPU/TPU

- Implementacja własnych warstw i operatorów dla frameworków ML
- Optymalizacja inferencji modeli w czasie rzeczywistym
- Przetwarzanie danych treningowych z wysoką wydajnością

### Przetwarzanie obrazów i wideo

- Filtrowanie i transformacja obrazów w czasie rzeczywistym
- Analiza wideo dla systemów wizji komputerowej
- Generowanie obrazów (np. w modelach generatywnych)
- Akceleracja algorytmów wykrywania i śledzenia obiektów

### Obliczenia numeryczne

- Symulacje fizyczne (dynamika płynów, mechanika kwantowa)
- Obliczenia macierzowe na dużą skalę
- Rozwiązywanie złożonych równań różniczkowych
- Analiza statystyczna dużych zbiorów danych

## Mojo w sztucznej inteligencji

Mojo został zaprojektowany z myślą o zastosowaniach AI i ML — zarówno do pisania wysokopoziomowej logiki, jak i optymalizacji niskopoziomowych kernelów dla akceleratorów (GPU, TPU, NPU).

Zalety:

- Bezpośrednia kontrola nad alokacją i przenoszeniem danych do akceleratorów
- Kompatybilność z bibliotekami Pythona (np. NumPy, TensorFlow)
- Możliwość pisania własnych operacji dla modeli AI, które są potem ekstremalnie szybkie

Przykładowe zastosowania:

- Pisanie niestandardowych warstw neuronowych
- Optymalizacja obliczeń macierzowych
- Inference modeli na edge-devices bez Pythonowego runtime'u

Mojo umożliwia np. stworzenie własnego kernelu do wykonania danego obliczenia na GPU działającego szybciej niż rozwiązania w bibliotekach, które nie oferują obliczeń na GPU.

## Wykonywanie obliczeń podczas kompilacji

Mojo udostępnia możliwość wykonywania fragmentów kodu podczas kompilacji i zapisywaniu wyniku obliczeń w pliku wykonywalnym, jest to podobny mechanizm do `constexpr` w C++, czy `comptime` w Zig'u. Służy do tego keyword `alias`.

```mojo
from memory import UnsafePointer, Pointer


fn fib(n: Int) -> Int32:
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


fn squared(n: Int) -> UnsafePointer[Int32]:
    var tmp = UnsafePointer[Int32].alloc(n)
    for i in range(n):
        tmp.store(i, fib(i))
    return tmp


def main():
    # alias: during comptime
    alias n_numbers = 34
    alias precaculated = squared(n_numbers)

    for i in range(n_numbers):
        print(precaculated.load(i))

    precaculated.free()
```

Bez aliasów kod wykonuje się w 50.8ms, podczas gdy z ich użyciem czas spada do 18.4ms. Za tę optymalizację płacimy jednak znacząco wydłużonym czasem kompilacji - z 2.2s aż do 24s - gdyż system metaprogramowania kompilacji w czasie wykonywania (comptime) Mojo nie jest jeszcze tak dobrze zoptymalizowany jak jego odpowiedniki w dojrzałych językach.

## Ownership system z Rusta

Jedną z najbardziej innowacyjnych cech Mojo jest wprowadzenie systemu własności (ownership) podobnego do tego znanego z języka Rust. System ten zapewnia bezpieczeństwo pamięci bez konieczności korzystania z garbage collectora, co jest kluczowe dla wysokowydajnych zastosowań.

Główne zasady systemu własności w Mojo:

- Każda wartość ma dokładnie jednego właściciela
- Kiedy właściciel wychodzi poza zakres, wartość jest automatycznie dealokowana
- Wartości mogą być pożyczone (borrowed) przez referencje, ale właściciel nie może ich modyfikować dopóki są pożyczone

Przykładowy kod demonstrujący system własności:

```py
from memory import UnsafePointer

fn borrow_example():
    var my_string = String("Hello Mojo")
    var ptr = UnsafePointer[String].alloc(1)

    # Przenieś własność my_string do ptr
    ptr.init_pointee_move(my_string^)

    # Próba użycia my_string tutaj spowodowałaby błąd kompilacji
    # ponieważ jego własność została przeniesiona
    # print(my_string)

    print(ptr[])  # Wypisuje: Hello Mojo

    # Zniszcz wartość, na która wskazuje ptr
    ptr.destroy_pointee()
    ptr.free()

fn main():
    borrow_example()
```

## Wskaźniki

Mojo, mimo swojej wysokopoziomowej składni, oferuje pełny dostęp do niskopoziomowych mechanizmów zarządzania pamięcią poprzez wskaźniki. Możliwość ta jest kluczowa dla optymalizacji wydajności w krytycznych fragmentach kodu.

### Mojo oferuje kilka rodzajów wskaźników:

- UnsafePointer - podstawowy wskaźnik bez żadnych zabezpieczeń
- Pointer - wskaźnik z podstawowymi zabezpieczeniami
- OwnedPointer - wskaźnik z automatycznym zarządzaniem czasem życia

Przykład użycia wskaźników:

```py
from memory import UnsafePointer, Pointer

fn unsafe_pointer_example():
    var size = 5
    var ptr = UnsafePointer[Int32].alloc(size)

    for i in range(size):
        ptr.store(i, i * 10)

    for i in range(size):
        print("Value at index", i, ":", ptr.load(i))

    ptr.free()

fn safe_pointer_example():
    var data = 42
    var ptr = Pointer.address_of(data)

    print(ptr[])

fn main():
    unsafe_pointer_example()
    safe_pointer_example()
```

Dzięki wskaźnikom Mojo pozwala na:

- Bezpośredni dostęp do pamięci akceleratorów (GPU, TPU)
- Optymalizację transferu danych między CPU a urządzeniami
- Tworzenie własnych struktur danych zoptymalizowanych pod kątem wydajności
- Integrację z bibliotekami napisanymi w C/C++

Wskaźniki w Mojo zapewniają elastyczność charakterystyczną dla języków niskopoziomowych, jednocześnie zachowując czytelność i bezpieczeństwo dzięki systemowi typów i właściwości.

## Zarządzanie pamięcią

W przeciwieństwie do Pythona, Mojo **nie posiada garbage collectora**. Zamiast tego, programista ma większą kontrolę nad zarządzaniem pamięcią, podobnie jak w C/C++ czy Rust.

Dzięki temu:

- Można pisać bardziej przewidywalny i wydajny kod
- Łatwiej jest debugować zachowania związane z pamięcią
- Oprogramowanie może być używane w systemach wbudowanych lub na urządzeniach o ograniczonych zasobach (IoT, edge AI)

To podejście idealnie sprawdza się w systemach, gdzie czas wykonania i deterministyczność mają kluczowe znaczenie — np. przy inferencji modeli AI w czasie rzeczywistym.

## Roadmapa Mojo i aktualny status

Mojo jest wciąż rozwijany i dostępny obecnie przez **Mojo Playground** – specjalne środowisko przeglądarkowe dostępne po zapisaniu się do testów (https://www.modular.com/mojo).

## Aktualne ograniczenia:

- Brak w pełni rozwiniętego systemu paczek
- Brak dokumentacji pełnej wersji języka (dostępny tylko subset)
- Język jest jeszcze w fazie testów i społeczność co chwile wprowadza niekompatybilne zmiany, przez co większość kodu/przykładów przestaje działać.

## Planowane funkcje:

- Integracja z popularnymi IDE jak VS Code czy PyCharm (VSCode już ma wtyczke Mojo)
- Integracja z istniejącymi projektami ML
- System paczek i menedżer zależności
- Obsługa większej liczby backendów sprzętowych (CPU, GPU, FPGA)

Mojo ma potencjał stać się "językiem Pythona 2.0" dla AI, oferując nowy standard w wydajności i skalowalności, jednocześnie zachowując prostotę, wprowadzając abstrakcje i optymalizacje pod różne akceleratory obliczeniowe. Nie da się ukryć, że jest to wciąż język w fazie intensywnego rozwoju. Deweloperzy Mojo ciągle eksperymentują ze składnią i różnymi podejściami do rozwiązywania problemów, przez co wiele rzeczy może się zmieniać z dnia na dzień.

