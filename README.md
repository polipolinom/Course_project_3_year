# Алгоритмы вычисления сингулярного разложения многоканальных сверток

Данный репозиторий содержит реализацию вычислительно-устойчивого алгоритма для подсчета сингулярного разложения матрицы сверточного слоя корреляции, а также алгоритм регуляризации сверточной нейронной сети на основе предложенного алгоритма.

## Сборка
Для использования данной библиотеки необходимо установить [данную библиотеку](https://github.com/polipolinom/SVD_course_project_2_year) и добавить в свой проект все header-файлы из `src\algorithms`

## Работа с библиотекой
`std::vector<long double> svd_convolution_1d(std::vector<std::vector<Matrix>>& kernels, size_t signal_size, Matrix* left_basis, Matrix* right_basis, size_t stride, bool full_basis, long double eps)` возвращает сингулярные числа свертки с ядром `kernels`, примененной к матрице размером $1 \times$ `signal_size`, со страйдом `stride`. Записывает полные базисы (если `full_basis` равно `True`) или базисы для восстановления ядра в `left_basis` и `right_basis`. Вычисления выполняются с точностью `eps`

`std::vector<long double> svd_convolution_2d(std::vector<std::vector<Matrix>>& kernels, size_t image_height, size_t image_width, Matrix* left_basis, Matrix* right_basis, size_t stride, bool full_basis, long double eps)` возвращает сингулярные числа свертки с ядром `kernels`, примененной к матрице размером `image_height` $\times$ `image_width`, со страйдом `stride`. Записывает полные базисы (если `full_basis` равно `True`) или базисы для восстановления ядра в `left_basis` и `right_basis`. Вычисления выполняются с точностью `eps`.

`std::vector<std::vector<Matrix>> clip_singular_1d(std::vector<std::vector<Matrix>>& kernels, size_t signal_size, size_t stride, long double lower_bound, long double upper_bound, long double eps)` возвращает ядро свертки `kernels`, примененняемого к матрице размером $1 \times$ `signal_size`, со страйдом `stride`, после клиппирования его сингулярных значений, ограничивая их на отрезок `[lower_bound, upper_bound]`. Вычисления выполняются с точностью `eps`.

`std::vector<std::vector<Matrix>> clip_singular_2d(std::vector<std::vector<Matrix>>& kernels, size_t image_height, size_t image_width, size_t stride, long double lower_bound, long double upper_bound, long double eps)` возвращает ядро свертки `kernels`, примененняемого к матрице размером `image_height` $\times$ `image_width`, со страйдом `stride`, после клиппирования его сингулярных значений, ограничивая их на отрезок `[lower_bound, upper_bound]`. Вычисления выполняются с точностью `eps`.

`std::vector<long double> svd_banded(Matrix A, size_t band_width, long double eps)` возвращает сингулярные числа ленточной верхнетругольной матрицы `A` с шириной ленты `band_width`, применяя неявный QR алгоритм для ленточных матриц. Вычисления выполняются с точностью `eps`.

`std::vector<long double> svd_banded_reduction(Matrix& A, Matrix* left_basis, Matrix* right_basis, long double eps)` возвращает сингулярные числа ленточной верхнетругольной матрицы `A`, которая записана в сжатом виде, применяя редукцию ленты. Вычисляет базисы `left_basis`, `right_basis`. Вычисления выполняются с точностью `eps`.
