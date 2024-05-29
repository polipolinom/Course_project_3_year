# Алгоритмы вычисления сингулярного разложения многоканальных сверток

Данный репозиторий содержит реализацию вычислительно-устойчивого алгоритма для подсчета сингулярного разложения матрицы сверточного слоя корреляции, а также алгоритм регуляризации сверточной нейронной сети на основе предложенного алгоритма.

## Сборка
Для использования данной библиотеки необходимо установить [данную библиотеку](https://github.com/polipolinom/SVD_course_project_2_year) и добавить в свой проект все header-файлы из `src\algorithms`

## Работа с библиотекой
`vector<long double> svd_convolution_1d(std::vector<std::vector<Matrix>>& kernels, size_t signal_size, Matrix* left_basis, Matrix* right_basis, size_t stride, bool full_basis, long double eps)` возвращает сингулярные числа свертки с ядром `kernels`, примененной к матрице размером $1 \times$ `signal_size`, со страйдом `stride`. Записывает полные базисы (если `full_basis` равно `True`) или базисы для восстановления ядра в `left_basis` и `right_basis`. Вычисления выполняются с точностью `eps`

    \item \texttt{vector svd\_convolution\_2d(std::vector<std::vector<Matrix>\! >\& kernels, \\ size\_t image\_height, size\_t image\_width, Matrix* left\_basis, \\ Matrix* right\_basis, size\_t stride, bool full\_basis, long double eps)}~--- возвращает сингулярные числа свертки с ядром \texttt{kernels}, примененной к матрице размером \texttt{image\_height} $\times$ \texttt{image\_width}, со страйдом \texttt{stride}. Записывает полные базисы (если \texttt{full\_basis} равно \texttt{True}) или базисы для восстановления ядра в \texttt{left\_basis} и \texttt{right\_basis}. Вычисления выполняются с точностью \texttt{eps}.

    \item \texttt{std::vector<std::vector<Matrix>\! > clip\_singular\_1d( \\std::vector<std::vector<Matrix>\! >\& kernels, size\_t signal\_size, \\ size\_t stride, long double lower\_bound, long double upper\_bound, \\long double eps)} ~--- возвращает ядро свертки \texttt{kernels}, примененняемого к матрице размером $1 \times$ \texttt{signal\_size}, со страйдом \texttt{stride}, после клиппирования его сингулярных значений, ограничивая их на отрезок \texttt{[lower\_bound, upper\_bound]}. Вычисления выполняются с точностью \texttt{eps}.

    \item \texttt{std::vector<std::vector<Matrix>\! > clip\_singular\_2d( \\std::vector<std::vector<Matrix>\! >\& kernels, size\_t image\_height, \\ size\_t image\_width, size\_t stride, long double lower\_bound, \\ long double upper\_bound, long double eps)} ~--- возвращает ядро свертки \texttt{kernels}, примененняемого к матрице размером \texttt{image\_height} $\times$ \texttt{image\_width}, со страйдом \texttt{stride}, после клиппирования его сингулярных значений, ограничивая их на отрезок \texttt{[lower\_bound, upper\_bound]}. Вычисления выполняются с точностью \texttt{eps}.

    \item \texttt{std::vector svd\_banded(Matrix A, size\_t band\_width, long double eps)} ~--- возвращает сингулярные числа ленточной верхнетругольной матрицы \texttt{A} с шириной ленты \texttt{band\_width}, применяя неявный QR алгоритм для ленточных матриц. Вычисления выполняются с точностью \texttt{eps}.

    \item \texttt{std::vector svd\_banded\_reduction(Matrix\& A, Matrix* left\_basis, \\ Matrix* right\_basis, long double eps)} ~--- возвращает сингулярные числа ленточной верхнетругольной матрицы \texttt{A}, которая записана в сжатом виде, применяя редукцию ленты. Вычисляет базисы \texttt{left\_basis, right\_basis}. Вычисления выполняются с точностью \texttt{eps}.
