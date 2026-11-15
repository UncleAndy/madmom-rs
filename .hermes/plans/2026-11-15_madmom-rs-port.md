# Портирование madmom (Python) → Rust: план реализации

> **Для исполнителя-агента:** план самодостаточен. Спека — это исходники Python
> в `third_party/madmom`; читать их рядом с планом. Шаги помечены `- [ ]`.
> Правки файлов делаем python-скриптом/heredoc (см. «Инварианты»), не `write_file`.

**Goal:** Довести `madmom-rs` до состояния, когда Rust-библиотека воспроизводит
все публичные процессоры madmom (beat/onset/tempo/downbeat/note/chord/key/chroma)
на предобученных моделях, с численной эквивалентностью в пределах заявленных
допусков и с CLI-программами, повторяющими `bin/*`.

**Architecture:** Cargo-workspace из 8 крейтов (core/audio/ml/features/io/eval)
+ фасад `madmom-rs` + `madmom-cli`. Python-граф процессоров (`Processor`, работающий
над `numpy.ndarray` и его подклассами) транслируется в типизированные Rust-структуры
с данными в `ndarray`. NN-инференс в Rust **не портируется**: активации считает
`tritonserver` (клиент — `tritonserver-rs`), Rust владеет пре-процессингом до входа
сети и всей постобработкой, а для локальной разработки и тестов Фаз 3–5 активации
берутся из фикстур madmom (golden-режим). Верификация — на «золотых» данных самого
madmom (`tests/data/activations/*.npz`, `tests/data/detections/*.txt`) и на
промежуточных дампах, снятых с собранного эталона.

**Tech Stack:** Rust (`ndarray`, `rustfft`/`realfft`, `tritonserver-rs`, `serde`/`serde_json`,
`rayon`, `clap`, `hound`, `byteorder`, `flate2`/`zip` для `.npz`, `thiserror`);
Python-инструменты (numpy, scipy, cython, pytest) через `nix-shell`;
ffmpeg для декодирования не-WAV.

**Spec:** `third_party/madmom/` (submodule CPJKU/madmom, версия `0.17.dev0`,
HEAD `27f032e`), модели — `third_party/madmom/madmom/models/` (вложенный сабмодуль,
CPJKU/madmom_models@7e3dc1b), фикстуры и тесты —
`third_party/madmom/tests/`.

## Global Constraints

- API-имена повторяют madmom: `RnnBeatProcessor` ≈ `RNNBeatProcessor`, имена полей
  соответствуют параметрам конструкторов Python. Расхождение допускается только
  явно и с комментарием `// differs from madmom: <причина>`.
- Комментарии, doc-комментарии, коммиты, README — на английском.
- Ни один тест не помечается `#[ignore]`; недоступный тест удаляется или
  помечается `#[cfg(feature = "...")]` с объяснением в комментарии.
- Модели не конвертируются и не вендорятся: NN-инференс идёт через `tritonserver`,
  в крейты попадает только контракт входа/выхода. Лицензия моделей (CC BY-NC-SA 4.0,
  NonCommercial) — забота того, кто их размещает в Triton-репозитории.
- Публичное имя крейта-фасада — `madmom-rs`; id пакетов CLI — `madmom-cli`.
- Правки любых файлов — python-скриптом или `cat <<'EOF'` через терминал
  (`write_file`/`patch` дописывают ESC/BEL и CRLF).
- Прогоны длиннее ~2 минут — `terminal(background=true, notify=true)`.
- Перед правкой/замером: `git status` → точная команда → на исполняемом ли пути
  находится правило. Только потом логика.

## Review Focus

Входы, на которых «разумный» порт молча расходится с madmom (каждый пункт
закрывается тестом в задаче, владеющей кодом):

1. **Не-WAV и не-44100 входы** (`stereo_sample.m4a`, `stereo_chirp_rg.flac`,
   `sample_22050.wav`): madmom уходит в ffmpeg-декодер целиком, включая
   ресемплинг и ReplayGain; наивный «свой wav-ридер» даст другой результат.
2. **Каналы**: `num_channels=1` + `channel=N` (брать один канал) против
   `channel=None` (усреднение каналов с приведением к исходному dtype) —
   `remix()` в `audio/signal.py`. Разные активации при одинаковом на вид входе.
3. **Короткие сигналы** (< `frame_size`): путь padding в `signal_frame()`,
   включая `pad='repeat'` и повтор первого/последнего сэмпла.
4. **Тишина и пустой вход**: `threshold_activations()` → пустой список биений,
   а не паника; нулевая длина аудио; `np.isinf` в Viterbi → предупреждение и
   пустой путь.
5. **Границы порогов**: пик-пикинг (`>= threshold`), `argrelmax(mode='wrap')`,
   `uniform_filter`/`maximum_filter` с `origin` — расхождение ровно на кадр
   меняет выход `.txt` на строку; нужна явная политика допуска (см. §4).

---

## 1. Что портируется — инвентарь (измерено на этом чек-ауте)

### 1.1 Исходники

| пакет | строк `.py` | файлов |
|---|---|---|
| `madmom/features` | 6695 | 10 |
| `madmom/audio` | 6272 | 8 |
| `madmom/evaluation` | 4447 | 7 |
| `madmom/utils` | 2685 | 3 |
| `madmom/ml` | 2401 | 6 |
| `madmom/io` | 1939 | 3 |
| `madmom/` (корень, `processors.py` и пр.) | 1147 | 3 |
| **библиотека всего** | **25 586** | **40** |
| Cython: `audio/comb_filters.pyx` (304), `features/beats_crf.pyx` (242), `ml/hmm.pyx` (753), `ml/nn/layers.pxd` (15) | 1314 | 4 |
| `tests/` | 12 740 | 33 (811 тестовых функций) |
| `bin/` | 25 программ (без расширения) | 25 |

Docstring-примеров (doctest, `>>>`) в библиотеке — **313**: это готовые численные
спецификации, их надо перенести в Rust-тесты.

### 1.2 Публичная поверхность, которую нужно воспроизвести

`madmom.processors` (8 классов: `Processor`, `OnlineProcessor`, `OutputProcessor`,
`SequentialProcessor`, `ParallelProcessor`, `IOProcessor`, `BufferProcessor`,
`_ParallelProcess`) + 6 функций; далее ~120 классов-процессоров в `audio`,
`features`, `ml`, `io`. Ключевые пайплайны (то, ради чего всё):

- `features.beats`: `RNNBeatProcessor`, `TCNBeatProcessor`, `BeatTrackingProcessor`,
  `BeatDetectionProcessor`, `CRFBeatDetectionProcessor`, `DBNBeatTrackingProcessor`,
  `MultiModelSelectionProcessor`.
- `features.onsets`: `SpectralOnsetProcessor` (11 ODF), `RNNOnsetProcessor`,
  `CNNOnsetProcessor`, `OnsetPeakPickingProcessor`, `peak_picking`.
- `features.tempo`: 5 гистограммных процессоров + `TempoEstimationProcessor`.
- `features.downbeats`: `RNNDownBeatProcessor`, `DBNDownBeatTrackingProcessor`,
  `PatternTrackingProcessor` (GMM), `RNNBarProcessor`, `DBNBarTrackingProcessor`,
  `LoadBeatsProcessor`, `SyncronizeFeaturesProcessor`.
- `features.notes`: `RNNPianoNoteProcessor`, `CNNPianoNoteProcessor`,
  `NoteOnsetPeakPickingProcessor`, `ADSRNoteTrackingProcessor` (+`features.notes_hmm`).
- `features.chords`: `CNNChordFeatureProcessor`, `CRFChordRecognitionProcessor`,
  `DeepChromaChordRecognitionProcessor`; `audio.chroma.DeepChromaProcessor`.
- `features.key`: `CNNKeyRecognitionProcessor` (24 класса тональностей).

### 1.3 Модели (`third_party/madmom/madmom/models`, 92 файла `.pkl`, ~35 МБ, ≈7.5 млн float-параметров)

> В Rust они **не переносятся**: конвертируются в ONNX (Task 0.8) и исполняются через
> Triton in-process. Таблица ниже —
> справка для того, кто будет собирать Triton-репозиторий (форматы, группы моделей,
> число параметров) и для контроля контракта входа/выхода в Task 2.2.

| каталог | файлов | параметров | слои, встречающиеся в файлах |
|---|---|---|---|
| `beats/2013` | 8 | 519 208 | BiLSTM (`BidirectionalLayer`+`LSTMLayer`+`Gate`+`Cell`) + `FeedForwardLayer` |
| `beats/2015` | 8 | 724 008 | то же (модели по умолчанию для offline-битов) |
| `beats/2016` | 8 | 237 608 | `LSTMLayer` + `FeedForwardLayer` (online) |
| `beats/2019` | 8 | 239 208 | `TCNLayer`/`TCNBlock`/`ConvolutionalLayer`/`MaxPoolLayer`/`AverageLayer`/`MultiTaskLayer`/`ReshapeLayer`/`SequentialLayer` |
| `downbeats/2016` | 20 | 1 170 636 | BiLSTM, `GRULayer`/`GRUCell` (rhythmic/harmonic) |
| `onsets/2013` | 17 | 541 193 | `RecurrentLayer`, BiLSTM, `ConvolutionalLayer`, `MaxPoolLayer`, `StrideLayer`, `BatchNormLayer` |
| `onsets/2014` | 8 | 1 288 | маленькие BiLSTM (post-processing сети для SuperFlux) |
| `notes/2013` | 1 | 209 352 | BiLSTM |
| `notes/2018` | 2 | 652 796 | CNN (`SequentialProcessor`+`ParallelProcessor`+`ConvolutionalLayer`+`BatchNormLayer`+`ReshapeLayer`+`FeedForwardLayer`) |
| `notes/2019` | 1 | 326 398 | CNN (`TransposeLayer` и др.) |
| `chords/2016` | 3 | 974 411 | `ConditionalRandomField`×2 + CNN-фичи |
| `chroma/2016` | 1 | 538 124 | `FeedForwardLayer`×4 |
| `key/2017` | 4 | 722 016 | CNN + `PadLayer`/`TransposeLayer`/`AverageLayer` |
| `key/2018` | 1 | 666 561 | CNN + `BatchNormLayer`/`MaxPoolLayer` |
| `patterns/2013` | 2 | — | `madmom.ml.gmm.GMM` (full covariance, для `GMMPatternTracker`) |

Полный список констант-реестра — `third_party/madmom/madmom/models/__init__.py`
(`BEATS_LSTM`, `BEATS_BLSTM`, `BEATS_TCN`, `DOWNBEATS_BLSTM`, `DOWNBEATS_BGRU`,
`ONSETS_RNN`, `ONSETS_BRNN`, `ONSETS_BRNN_PP`, `ONSETS_CNN`, `NOTES_BRNN`,
`NOTES_CNN`, `NOTES_CNN_MIREX`, `CHROMA_DNN`, `CHORDS_DCCRF`, `CHORDS_CNN_FEAT`,
`CHORDS_CFCRF`, `KEY_CNN`, `PATTERNS_BALLROOM`).

### 1.4 Фикстуры и тесты (готовая база для паритета)

- `tests/data/audio/` — 9 файлов: `sample.wav`, `sample2.wav`, `sample_22050.wav`,
  `stereo_sample.wav/.flac/.m4a/_rg.flac`, `stereo_chirp.wav/_rg.flac`.
- `tests/data/activations/` — 24 `.npz` (по ключам `activations` (float32) и
  `fps` (int64; местами 0-d object → нужен `allow_pickle`-совместимый читатель)):
  ожидаемые активации для beat/onset/note/chord/chroma/key пайплайнов.
- `tests/data/detections/` — 27 `.txt`: ожидаемые выходы 25 CLI-программ
  (биения, онсеты, аккорды, тональности, темпы, паттерны).
- `tests/data/annotations/` — 14 файлов (аннотации + MIDI) для модуля evaluation.
- `tests/*.py` — 811 тестов, из них 105 — end-to-end тесты CLI (`tests/test_bin.py`).

## 2. Ключевые решения

**D1. NN-слои в Rust не портируются: инференс — Triton in-process, модели — ONNX.**
Решение владельца: модели переписываем не руками, а **переносим в ONNX**; запуск —
`tritonserver-rs` 0.4.1 (локальный чек-аут `projects/RUST/tritonserver-rs`, MIT). Это
bindings к C-API Triton: сервер поднимается **в том же процессе** и линкуется с
`libtritonserver.so` — не сетевой клиент. Проверенные факты по крейту:
- репозиторий моделей: `models/<name>/config.pbtxt` + `models/<name>/<version>/model.onnx`;
- API (по исходникам `src/{options,server,request,response,memory}.rs`):
  `Options::new(repo)` → `.backend_directory("/opt/tritonserver/backends")`,
  `.model_control_mode(Control::Explicit)`, `.startup_model(name)`,
  `.strict_model_config(true)`, `.exit_timeout(..)`,
  `.pinned_memory_pool_byte_size(0)` для CPU-only; `Server::new(opts).await`
  (async, tokio) → `server.create_request(model, version)` (`-1` = последняя) →
  `request.add_default_allocator().add_input(name, Buffer)` → `request.infer_async()?.await?`
  → `Response::{get_outputs, get_output, return_buffers}`;
- фича `gpu` включена по умолчанию (вызывает `init_cuda()`); CPU-only =
  `default-features = false` + `instance_group` `KIND_CPU` в `config.pbtxt`;
- сборка/рантайм требуют `libtritonserver.so`, бинарников бэкендов (ONNX Runtime) и
  `libclang` для bindgen; минимальная версия контейнера — 24.08 (C-API 1.34); см. R16.

Следствия:
- из объёма порта выпадает `madmom/ml/nn/*` (1454 строки слоёв + `activations.py` +
  Cython-компиляция `layers.py`) — вместе с самым неблагодарным участком (эмуляция
  `scipy.ndimage.convolve` с `_kernel_margins`, `maximum_filter`, TCN-дилатации через
  `as_strided`): риски R2/R3 по NN-части закрываются сами;
- вся ответственность за точность переносится на **пре-процессинг** (Rust, Task 2.2)
  и на ONNX-перенос моделей (делает владелец; контракт входа/выхода — наш);
- NN-стадии остаются локальными, но появляется `Result` внутри синхронного API:
  вызов асинхронный (`infer_async`), наружу — синхронный фасад на внутреннем
  tokio-рантайме (`block_on`), чтобы не менять стиль API madmom; конфиг —
  `madmom.toml` / `MADMOM_MODEL_REPO` (D10); online-режим получает бюджет задержки (R14);
- локальные тесты Фаз 3–5 идут в **golden-режиме** (активации из фикстур madmom),
  поэтому Triton нужен только для Task 2.5.

*Справка для ONNX-переноса моделей (проверено на этом чек-ауте; сама конвертация —
не задача порта):* все 92 `.pkl` — pickle протокола 2 (Python 2) с объектами
`madmom.ml.nn.*`, `madmom.ml.gmm.GMM`, `madmom.processors.SequentialProcessor` и
`numpy.ndarray` внутри; читаются на Python 3.13 + NumPy 2.4 (92/92) с двумя поправками:
`notes/2018/notes_cnn_[12].pkl` ссылаются на исчезнувший в NumPy 2 `numpy.lib.shape_base`
(нужен модуль-шим), а переопределение `Unpickler.find_class` целиком ломает 11 файлов
(TCN/notes CNN) — теряется py2→py3 мэппинг `c_codecs`→`codecs`
(`Cannot interpret '<encode>' as a data type`); сначала `super().find_class()`, подмены
только для `madmom.*`. Не-NN модели тоже уезжают в ONNX (решение 3): `chords_dccrf/cnncrf`
— `ConditionalRandomField` с матрицами (`pi/tau/c/A/W`), `patterns/*ballroom_pattern_*.pkl`
— `GMM` (`weights/means/covars`, full covariance); обёртка для CRF: вход — наблюдения,
выход — последовательность состояний/меток; для GMM: вход — признаки кадра, выход —
log-плотности по компонентам.

**D2. Граница: Rust владеет пре-процессингом и постобработкой, ONNX-модели — сетями.**
Контракт: Rust отдаёт ровно тот тензор, который madmom подавал в первую сеть ансамбля
(те же разрешения, порядок `hstack`/`dstack`, паддинги), и ожидает ровно тот тензор,
который сеть возвращала. Усреднение ансамбля, `squeeze` и выбор выхода multi-task сети
(`itemgetter(0)`) остаются в Rust. Вход сверяется с дампами эталона
(`tests/golden/preproc/*.npy`), выход — с `.npz`-фикстурами.
Туда же уезжают две не-NN модели (решение 3): CRF аккордов вызывается batch-ом на файл,
GMM паттернов возвращает log-плотности по кадрам, которые подставляются в HMM как
`ObservationSource`. Математика `madmom.ml.gmm`/`madmom.ml.crf` в Rust не портируется.

**D3. Python-объекты-процессоры не портируются как механизм.**
`Processor.load/dump`, `pickle_processor`, `bin/PickleProcessor`, `io_arguments`
(аргументы `pickle`/`single`/`batch`/`online`) — это сериализация Python-объектов.
В Rust конфигурация пайплайна — типизированные структуры + `clap`; режимы
`single/batch/online` сохраняются как подкоманды CLI.

**D4. Подклассы `np.ndarray` → структуры «данные + метаданные».**
`Signal`, `Spectrogram`, `FilteredSpectrogram`, `LogarithmicSpectrogram`,
`SpectrogramDifference`, `ShortTimeFourierTransform`, `Filterbank`, `Filter`,
`Cepstrogram`, `MFCC`, `CLPChroma`, `Activations` — это ndarray-подклассы с
дополнительными атрибутами (`stft`, `filterbank`, `fps`, `sample_rate`, `bin_frequencies`).
В Rust каждый становится `struct`, который владеет `Array2<f32>`/`Array1<f64>`
и метаданными; цепочки — типизированные (см. §3.3). Динамическая композиция
нужна только CLI-слою.

**D5. Численные типы и порядок операций сохраняются буквально.**
- Аудио/спектр: `f64` там, где Python использует `float64` (окна, фильтрбанки
  `FILTER_DTYPE=float32`, STFT → `complex64` — `STFT_DTYPE=np.complex64`).
- Окно БПФ масштабируется в `ShortTimeFourierTransform`: при целочисленном сигнале
  `fft_window = window / np.iinfo(dtype).max` (т.е. `/32767` для int16). Это
  обязательный шаг паритета (иначе амплитуды спектрограмм отличаются в 32767 раз).
- NN-слои считают в `float32` (`NN_DTYPE=np.float32`), активации — как в
  `ml/nn/activations.py` (`sigmoid` через `expit`, `tanh`, `relu`, `elu`, `softmax`,
  `linear`), включая «in-place» семантику `out=`.
- Сигнал по умолчанию остаётся в исходном dtype (int16 для WAV); приведение к
  float происходит внутри пайплайнов; ресемплинг — только через ffmpeg.

**D6. `scipy`-поверхность воспроизводится точечно.** Точный список вызовов,
которые обязаны совпасть семантически (и по ним нужно написать тесты):

| место | вызов | замена в Rust |
|---|---|---|
| `audio/stft.py` | `scipy.fftpack.fft` (полный комплексный FFT, не rfft), `pyfftw` (опц.) | `rustfft` (complex), окно как в madmom |
| `audio/spectrogram.py` | `scipy.ndimage.maximum_filter` (tuning), `scipy.signal.filtfilt` (`SemitoneBandpassSpectrogram`) | фильтр скользящего максимума; `filtfilt` (b, a) — низкий приоритет |
| `audio/filters.py` | `scipy.signal.ellip` (`SemitoneBandpassFilterbank`) | реализация эллиптического фильтра — низкий приоритет |
| `audio/signal.py` | `scipy.signal.convolve2d` (2-D `smooth`) | `convolve2d` с `mode='same'` |
| `audio/hpss.py` | `scipy.ndimage.filters.median_filter` | медианный фильтр |
| `audio/cepstrogram.py` | `scipy.fftpack.dct` | DCT-II |
| `audio/io` | `scipy.io.wavfile` (запись WAV) | `hound` |
| `ml/nn/layers.py` | `scipy.ndimage.convolve` (mode='constant') + `_kernel_margins` (сдвиг амплитуд для чётных ядер), `maximum_filter` | собственная 2-D свёртка с нулевым заполнением и той же обрезкой полей |
| `ml/nn/activations.py` | `scipy.special.expit` | сигмоида (`0.5*(1+tanh(x/2))` — как fallback в madmom) |
| `ml/gmm.py` | `scipy.linalg.eigh` (`pinvh`) — только для обучения | не нужно (инференс: `log_multivariate_normal_density` = чистый numpy) |
| `ml/hmm.pyx` | `scipy.sparse.csr_matrix` в `make_dense/make_sparse` | своя CSR-сборка (`make_sparse` = группировка по prev_state + сортировка) |
| `features/onsets.py` | `maximum_filter`, `minimum_filter`, `uniform_filter` с `origin` | скользящие max/min/mean с тем же `origin` и `mode='constant'` |
| `features/tempo.py` | `scipy.signal.argrelmax(mode='wrap')`, `scipy.interpolate.interp1d('quadratic')` | локальные максимумы с обёрткой; квадратичная интерполяция |
| `features/beats.py` | `scipy.signal.argrelmin(mode='wrap')` | то же |
| `features/beats_crf.pyx` | `scipy.ndimage.correlate1d`, `scipy.stats.norm.pdf` | корреляция 1-D; нормальный PDF |
| `utils/stats.py` | `scipy.stats.chi2` | критерий МакНемара (только evaluation) |

**D7. HMM — ядро на Rust без Cython.** `ml/hmm.pyx` = CSR-матрица переходов
(`states`, `pointers`, `probabilities`), лог-доменный Viterbi с back-pointers,
нормализованный `forward`, потоковый `forward_generator` (блоками по
`block_size`). Порт в `madmom-ml::hmm` без потерь: тест — doctest из `hmm.pyx`
(два состояния, 9 наблюдений: путь `[1,1,0,0,1,1,1,0,0]`, `log_p=-12.87...`,
и таблица `forward`, приведённая в докстринге).

**D8. Таймстемпы и единицы измерения.** Внутри — кадры (`i64`), наружу —
секунды (`f64`), как в madmom (`activations / fps`). `fps=100` по умолчанию для
onsets, `100` для DBN-битов (`FRAME_SIZE=2048, HOP_SIZE=441` → 100 fps для 44100).

**D9. Параллелизм — `rayon`.** `ParallelProcessor` (multi-resolution спектрограммы)
и `NeuralNetworkEnsemble` (8 сетей) в madmom — `multiprocessing`/`num_threads`;
в Rust — пул `rayon` с детерминированным порядком слияния (`np.hstack`/average в
фиксированном порядке), иначе паритет плавает. 8 сетей ансамбля уходят отдельными
запросами к Triton (параллельно), но усреднение — строго в порядке madmom.

**D10. Triton — внешние конфиг и бинарники.** `madmom.toml`:
`[triton] repository = "models"`, `backend_directory = "/opt/tritonserver/backends"`,
`model_version = -1`, `control = "explicit"`, `cpu_only = true` (решение владельца:
пока только CPU-only); переопределения —
`MADMOM_MODEL_REPO` (по умолчанию `third_party/madmom-models-onnx/models`, см. §2.1),
`MADMOM_TRITON_BACKENDS`. Сервер поднимается один раз на процесс,
модели загружаются через `model_control_mode(EXPLICIT)` + `load_model`. Отсутствие
репозитория или либы даёт понятную ошибку при выборе `ActivationSource::TritonLocal`,
а не при старте и не в golden-режиме. Лицензия моделей (CC BY-NC-SA 4.0, NonCommercial) —
на стороне ONNX-репозитория; в крейты не попадает ничего, кроме контракта.

**D11. WAV-чтение — своё, всё остальное — ffmpeg.** Как в madmom:
`load_wave_file` (только PCM, через `wave`) → используется, если не требуется
ресемплинг/смена числа каналов; иначе `ffmpeg` (аргументы из
`io/audio.py::_ffmpeg_call`, включая `-af volume=replaygain=...`). Тесты
`tests/test_io_audio.py` переносятся 1:1 (в т.ч. ошибка для `foo_bar.wav`).

**D12. «Не портируем» по списку** — см. §7.

### 2.1 Спецификация ONNX-репозитория (решение принято, п. 1 прошлого доклада)

Раскладка (совместима с `config.pbtxt`-моделью Triton, см. D1):

```
third_party/madmom-models-onnx/          # отдельный репозиторий (позже — сабмодуль), CC BY-NC-SA
└── models/
    ├── beats_blstm_1/{config.pbtxt, 1/model.onnx}      # …_2 … _8 (2015)
    ├── beats_lstm_1/…_8            # 2016 (online)
    ├── beats_tcn_1/…_8             # 2019, два выхода
    ├── downbeats_blstm_1/…_8, downbeats_bgru_rhythmic_0…5, downbeats_bgru_harmonic_0…5
    ├── onsets_rnn_1/…_8, onsets_brnn_1/…_8, onsets_brnn_pp_1/…_8, onsets_cnn
    ├── notes_brnn, notes_cnn_1, notes_cnn_2, notes_cnn
    ├── chroma_dnn, key_cnn, key_cnn_1…4
    ├── chords_cnnfeat, chords_cnncrf, chords_dccrf
    └── ballroom_pattern_3_4, ballroom_pattern_4_4        # GMM
```

- имя каталога = stem исходного `.pkl` (источник — `third_party/madmom/madmom/models/`,
  92 файла уже выкачаны); версия — `1`; файл — `model.onnx`.
- **ансамбли — восемь отдельных моделей** (`beats_blstm_1…_8`), а не одна ансамблевая:
  так сохраняется паритет с madmom (усреднение 8 сетей делаем мы, D9) и появляется
  возможность сверить каждую сеть по отдельности с эталонным дампом.
- **имена тензоров:** вход — `input` (f32); выход — `output` (f32). Исключения:
  `beats_tcn_*` — `output_beats`, `output_tempo` (multi-task); GMM — вход `observation`,
  выход `log_densities`; CRF — вход `observation`, выходы `emission`, `transition`,
  `initial`, `final`.
- `config.pbtxt` (шаблон; CPU-only — решение владельца):
  ```textproto
  name: "beats_blstm_1"
  backend: "onnxruntime"
  max_batch_size: 0                     # батчинг по кадрам делает Rust, не Triton
  instance_group [{ kind: KIND_CPU }]
  input  [{ name: "input"  data_type: TYPE_FP32 dims: [-1, 162] }]
  output [{ name: "output" data_type: TYPE_FP32 dims: [-1, 1] }]
  ```
  (`-1` — переменное число кадров; для CNN/TCN вход 3-D: `dims: [-1, <freq>, <channels>]`.)
- путь по умолчанию: `MADMOM_MODEL_REPO` → `third_party/madmom-models-onnx/models` →
  `<repo>/models` (gitignored). Пока репозитория нет — Фазы 3–5 живут в golden-режиме.
- **скрипт конвертации — наш**, `tools/convert_to_onnx.py` (исполняемая спецификация;
  владелец может переиспользовать). Приёмка каждой модели — Task 2.4.

**Отображение слоёв madmom → ONNX (что обязан сделать конвертер).** Ловушки, на которых
расходится паритет, вынесены явно:

| madmom (`ml/nn/layers.py`) | ONNX | ловушка |
|---|---|---|
| `FeedForwardLayer` | `MatMul` + `Add` (+ активация) | — |
| `RecurrentLayer` | развёртка по кадрам или `Loop` | состояние `_prev`/`init` — начальное состояние в графе |
| `LSTMLayer` (4× `Gate`/`Cell`, **пипхолы**) | **не `LSTM`**: `MatMul`/`Add`/`Sigmoid`/`Tanh`/`Mul` с явной рекуррентностью | порядок гейтов ONNX — **i, o, f, c**, у madmom — input/forget/cell/output; пипхолы (`P`) в ORT не работают — см. блок ниже |
| `GRULayer` (`GRUCell`) | `GRU` (`linear_before_reset=1`) | madmom `out = ug*cell + (1-ug)*prev`, ONNX `H = (1-z)*h + z*H_prev` → гейт **инвертирован**; сверить по дампу |
| `BidirectionalLayer` | два направления + `Concat` | порядок конкатенации fwd/bwd |
| `ConvolutionalLayer` | `Pad` (constant) + `Conv` | обрезка полей по `_kernel_margins` (чётные ядра сдвигаются); madmom считает в (time, freq, chan), ONNX `Conv` — (N, C, H, W) → `Transpose` вокруг |
| `MaxPoolLayer` | `MaxPool` + срезы `[size//2 : n-(size-1)//2 : stride]` | та же семантика срезов, `mode='constant'` |
| `BatchNormLayer` | `Mul` + `Add` (инференс: `(x-mean)*(gamma*inv_std)+beta`) | — |

**ЛОВУШКА (измерено 2026-11-15, меняет предпосылку выше): пипхолы в LSTM/BLSTM.**

- у `blstm` (24 файла) максимум |p| = **35.1**, у `lstm` (8 файлов) — **2.80**;
  пипхолов нет у 60 моделей (`brnn` 17, `bgru` 12, `cnn` 10, `tcn` 8, `rnn` 8, `crf` 2,
  `gmm` 2, `feedforward` 1) — там `GRU`/`RNN`-узлы остаются законными;
- ONNX даёт у `LSTM` необязательный вход `P`, но **onnxruntime его не реализует**: узел с `P`
  не грузится (`INVALID_GRAPH: Type Error: Type 'tensor(float)' of input parameter (P) ... invalid`),
  а тот же узел без `P` считается. Опыт поставлен на onnxruntime 1.24.4, оба варианта;
- значит `LSTM`-узлом эти 32 модели выразить нельзя: молча выброшенные пипхолы дают модель,
  которая **считается и врёт** — именно тот отказ, который проект запрещает. Рекуррентность
  (i/f/o с `+p⊙c`, `g = tanh`, `c' = f⊙c + i⊙g`, `h' = o⊙tanh(c')`) разворачивается
  примитивами; `tools/onnx_spec.py::_lstm_gate_pack`/`_lstm_peepholes` уже готовят тензоры
  (порядок i,o,f,c и пипхолы в порядке ONNX), нужен сам граф;
- старая реализация `build_lstm` вдобавок паковала узлы неверно (`B` попадал в слот `P`,
  смещения гейтов обнулялись) — теперь она **отказывается** строить граф (`UnsupportedLayer`
  с причиной), вместо того чтобы выдать неправильную модель.

**РЕШЕНИЕ (владелец, 2026-11-15): вариант B.** Рекуррентность LSTM/BLSTM строится
примитивами (`Scan`/`Loop` с переносимым состоянием `(h, c)`, тело цикла —
`MatMul`/`Add`/`Mul`/`Sigmoid`/`Tanh` + `Slice` по гейтам), порядок операций — точно как в
`Gate.activate`/`LSTMLayer.activate` (пипхол **до** рекуррентного вклада; гейт `o` смотрит
на **новое** `c_t`, гейты `i`/`f` — на `c_{t-1}`; `c_t = g⊙i + c_{t-1}⊙f`). Вариант C
(LSTM в Rust) отклонён: архитектура «весь NN — через ONNX/Triton» сохраняется, у
верификации остаётся одна точка — `tools/validate_onnx.py`. Проверять обязательно на
полной последовательности (281 кадр): рекуррентность накапливает разницу порядков
сложений float32.
| `PadLayer`/`TransposeLayer`/`ReshapeLayer`/`AverageLayer` | `Pad`/`Transpose`/`Reshape`/`ReduceMean` | `keepdims`, порядок осей |
| `MultiTaskLayer`/`SequentialLayer`/`ParallelLayer` | структура графа (разные входы/выходы) | выбор выхода (`itemgetter(0)`) остаётся в Rust |
| `TCNBlock`/`TCNLayer` | `Conv` с `dilations`, нулевой `Pad` `2*dilation_rate`, `Add` (residual), `Concat` (skip) | дилатация у madmom сделана через `as_strided` — сверить с `Conv(dilations=…)` |

### 2.2 Контракт ONNX для CRF и GMM (решение принято, п. 2 прошлого доклада)

- **GMM** (`ballroom_pattern_3_4.pkl`, `ballroom_pattern_4_4.pkl`): модель целиком
  уезжает в ONNX. Вход `observation` f32 `[frames, num_features]`; выход
  `log_densities` f32 `[frames, num_densities]`, где `num_densities = 2 блока × число
  GMM` и порядок столбцов — ровно тот, который задаёт
  `GMMPatternTrackingObservationModel.pointers` (Rust подставляет значения в HMM как
  `ObservationSource`, ничего не считая). Формула внутри графа:
  `log N(x | μ, Σ) + log w`, `covariance_type='full'`.
- **CRF** (`chords_dccrf.pkl`, `chords_cnncrf.pkl`): модель (параметры + скоринг) —
  в ONNX, но **не** динамическое программирование. Граф отдаёт
  `emission` f32 `[frames, states]`, `transition` f32 `[states, states]`,
  `initial` f32 `[states]`, `final` f32 `[states]`; Viterbi выполняет Rust
  (`madmom_features::chords`, ~30 строк — перенос `ml/crf.py::process`, тест —
  `tests/test_ml_crf.py`).
  *Почему так:* последовательный Viterbi в графе требует `Loop`/`Scan` с carried-state
  и ручного back-pointer тензора — хрупко и не проверяемо по частям; сам DP — это
  обычный код, как Viterbi в `hmm.pyx`, который мы всё равно портируем. Это осознанное
  уточнение решения 3: в ONNX уходит *модель*, а не общий алгоритм.

## 3. Целевое устройство

### 3.1 Крейты

```
madmom-rs/
├── Cargo.toml                 # workspace
├── flake.nix                  # devShell: rust + python(numpy/scipy/cython/pytest) + ffmpeg
├── crates/
│   ├── madmom-core/           # Processor/Online/Sequential/Parallel/IO/Buffer, utils
│   ├── madmom-audio/          # signal, stft, spectrogram, filters, chroma, cepstrogram, hpss, comb_filters
│   ├── madmom-ml/             # hmm, inference (golden-фикстуры + triton in-process/ONNX)
│   ├── madmom-features/       # activations, onsets, beats, beats_hmm, beats_crf, downbeats, tempo, notes(+notes_hmm), chords, key
│   ├── madmom-io/             # audio (wav/ffmpeg), events(txt), npy/npz, midi
│   ├── madmom-eval/           # evaluation: onsets/beats/tempo/notes/chords/key
│   ├── madmom/                # фасад: re-export + готовые пайплайны (BeatTracker и т.п.)
│   └── madmom-cli/           # 24 бинаря в v1 (bin/*, кроме evaluate/PickleProcessor)
├── tools/
│   ├── dump_reference.py      # промежуточные эталонные дампы с собранного madmom
│   ├── dump_preproc.py        # входы/выходы сетей эталона (граница локально/удалённо)
│   └── build_reference.sh     # nix-shell + setup.py build_ext --inplace + pytest
├── madmom.toml                # [triton] url, имена/версии моделей и тензоров
└── tests/                     # интеграционные тесты паритета (Rust)
    ├── fixtures/              # симлинки/копии third_party/madmom/tests/data
    └── golden/                # дампы эталона по стадиям
```

### 3.2 Соответствие модулей

| Python | Rust |
|---|---|
| `madmom.processors` | `madmom_core::processors` |
| `madmom.utils` (`segment_axis`, `combine_events`, `quantize_events`, `expand_notes`, `search_files`) | `madmom_core::utils` |
| `madmom.audio.signal` | `madmom_audio::signal` |
| `madmom.audio.stft` | `madmom_audio::stft` |
| `madmom.audio.spectrogram` | `madmom_audio::spectrogram` |
| `madmom.audio.filters` | `madmom_audio::filters` |
| `madmom.audio.{chroma,cepstrogram,hpss,comb_filters}` | `madmom_audio::{chroma,cepstrogram,hpss,comb_filters}` |
| `madmom.ml.hmm` | `madmom_ml::hmm` (`.nn`/`.nn.layers` — не портируются, `.gmm`/`.crf` — ONNX) |
| `madmom.models` (реестр) | `madmom_ml::models` |
| `madmom.features` (`Activations`) | `madmom_features::activations` |
| `madmom.features.{onsets,beats,beats_hmm,beats_crf,downbeats,tempo,notes,notes_hmm,chords,key}` | соответствующие модули `madmom_features::*` |
| `madmom.io.{audio,__init__,midi}` | `madmom_io::{audio,events,midi}` (+`npy`) |
| `madmom.evaluation.*` | `madmom_eval::*` |
| `bin/*` (25 программ) | `madmom-cli` (`[[bin]]` × 24 в v1; `evaluate` отложен) |

### 3.3 Ключевые интерфейсы (это то, вокруг чего строится всё остальное)

```rust
// madmom-core
pub trait Processor { type Input; type Output; fn process(&mut self, input: Self::Input) -> Self::Output; }
pub trait OnlineProcessor: Processor { fn reset(&mut self); fn process_online(&mut self, d: Self::Input) -> Self::Output; }
pub struct Sequential<P: Processor>(pub Vec<Box<dyn /* типизированная цепочка */>>); // см. ниже
pub struct BufferProcessor { pub buffer_size: Vec<usize>, pub init: ArrayD<f32>, data: ArrayD<f32> }
  impl BufferProcessor { pub fn process(&mut self, data: &ArrayD<f32>) -> ArrayD<f32>; } // np.roll-семантика

// madmom-audio
pub struct Signal { pub data: Array2<f32>, pub sample_rate: Option<u32>, pub num_channels: usize,
                    pub start: Option<f64>, pub stop: Option<f64> }
pub fn signal_frame(signal: &Array2<f32>, index: i64, frame_size: usize, hop_size: f64,
                    origin: i64, pad: PadMode) -> Array2<f32>;
pub struct FramedSignal { /* frame_size, hop_size (f64!), origin (i64), num_frames, end */ }
pub fn stft(frames: &Array2<f32>, window: Option<&Array1<f64>>, fft_size: Option<usize>,
            circular_shift: bool, include_nyquist: bool) -> Array2<Complex32>;
pub struct Filterbank { pub filters: Array2<f32>, pub start: usize, pub stop: usize,
                        pub center_frequencies: Vec<f64> }
pub fn frequencies2bins(freqs: &[f64], bin_freqs: &[f64], unique: bool) -> Vec<usize>; // searchsorted+clip+closest

// madmom-ml
pub struct TransitionModel { pub states: Vec<u32>, pub pointers: Vec<u32>, pub probabilities: Vec<f64> }
pub trait ObservationModel { fn pointers(&self) -> &[u32];
    fn log_densities(&self, obs: &Array2<f64>) -> Array2<f64>;
    fn densities(&self, obs: &Array2<f64>) -> Array2<f64> { self.log_densities(obs).mapv(f64::exp) } }
pub struct HiddenMarkovModel { tm: TransitionModel, om: Box<dyn ObservationModel>, initial: Vec<f64>, prev: Vec<f64> }
  impl HiddenMarkovModel {
      pub fn viterbi(&mut self, obs: &[f64]) -> (Vec<u32>, f64);          // лог-домен + back-pointers
      pub fn forward(&mut self, obs: &Array2<f64>, reset: bool) -> Array2<f64>; // нормализация по кадру
      pub fn forward_step(&mut self, obs: &[f64]) -> Vec<f64>;             // online (forward_generator)
  }
// Инференс: golden-фикстуры (разработка/тесты) или Triton in-process (ONNX). Слои не реализуются (D1).
pub enum InputShape { Frames(usize), FramesFreq(usize, usize), FramesFreqCh(usize, usize, usize) }
pub trait ActivationModel { fn input_shape(&self) -> InputShape;
                            fn predict(&mut self, f: &ArrayD<f32>) -> Result<ArrayD<f32>>; }
pub struct TritonModel {                     // sync-фасад над infer_async (D1)
    pub server: Arc<Server>, pub model_name: String, pub model_version: i64,
    pub input_name: String, pub output_name: String, pub shape: InputShape,
    pub runtime: tokio::runtime::Runtime,
}
pub trait ObservationSource { fn log_densities(&self, o: &Array2<f64>) -> Result<Array2<f64>>; }  // GMM → ONNX
pub struct NeuralNetworkEnsemble { models: Vec<Box<dyn ActivationModel>>, ensemble: EnsembleFn }
  impl NeuralNetworkEnsemble { pub fn load(spec: &EnsembleSpec) -> Result<Self>; }

// madmom-io
pub fn load_audio_file(p: &Path, sample_rate: Option<u32>, num_channels: Option<usize>,
                       channel: Option<usize>, start: Option<f64>, stop: Option<f64>,
                       dtype: Option<DType>) -> Result<(Array2<f32>, Option<u32>)>;
pub fn load_wave_file(p: &Path, ...) -> Result<(Array2<f32>, Option<u32>)>;   // только PCM
pub struct Activations { pub data: ArrayD<f32>, pub fps: Option<f64> }       // .npz / текст / ndarray
```

Замечание по `Sequential`: madmom-цепочки меняют тип данных между стадиями
(`Signal`→`FramedSignal`→`STFT`→`Spectrogram`→`FilteredSpectrogram`→`…`).
Поэтому в библиотеке используется типизированная композиция (`Sequence<A,B>` со
стадиями-функциями `FnMut(In) -> Out`), а динамический `enum Stage` вводится
только для CLI, где цепочка собирается из аргументов.

## 4. Верификация

### 4.1 Четыре источника истины

1. **Численные ассерты тестов madmom** (`tests/test_*.py`, 811 тестов) — переносятся
   в Rust 1:1 (значения уже посчитаны эталоном).
2. **Готовые фикстуры** — `tests/data/activations/*.npz` (ожидаемые активации) и
   `tests/data/detections/*.txt` (ожидаемые выходы 25 программ). Позволяют
   проверять пайплайн целиком без запуска Python.
3. **Дампы промежуточных стадий** с собранного эталона (`tools/dump_reference.py`):
   для каждого шага пайплайна (signal → frames → stft → spec → filt → log → diff →
   activations → path → detections) сохраняется массив в `.npy`, чтобы ловить
   ошибку на первой стадии, а не на последней.
4. **Triton in-process** (feature `triton-live`, Task 2.5): если активации из ONNX не
   совпали с `.npz`, дефект ищется сначала в пре-процессинге (`golden/preproc/*.npy`),
   потом в ONNX-переносе модели — но это не повод менять допуск.

### 4.2 Допуски (пороги задаются ДО прогонов, менять их после получения данных нельзя)

| слой | метрика | допуск |
|---|---|---|
| Аудио/окна/фильтрбанки (float64) | `max|Δ|` / относительная | `≤1e-9` отн. / `≤1e-6` абс. |
| STFT/Spectrogram (complex64/float32) | `max|Δ|` | `≤1e-5` абс. |
| NN-активации (float32, другой порядок суммирования) | `max|Δ|`, косинус | `≤1e-4` абс., `≥0.99999` |
| Viterbi-путь / пик-пикинг / детекции | индексы кадров | совпадение множеств, допускается расхождение ≤1 кадра лишь если причина зафиксирована в комментарии теста |
| Списки биений/онсетов в секундах | `max|Δ|` | `≤1e-6` |
| Метрики evaluation | `max|Δ|` | `≤1e-9` |

Правило: если расхождение выше допуска — сначала проверить, не нарушен ли
порядок операций (D5/D6), и только потом менять допуск. Изменение допуска
без объяснения = подгонка под результат.

### 4.3 Гейты

- `cargo test --workspace` — юниты + паритет-тесты (фикстуры) + doc-тесты.
- `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`.
- `tools/build_reference.sh && nix-shell … 'python3 -m pytest tests -q'` —
  фиксируем эталонный baseline (сколько тестов madmom проходит на этой машине
  до порта; это обязательный шаг Фазы 0, потому что часть тестов требует
  ffmpeg/opencv/pyfftw).
- Отдельный «end-to-end» гейт: `madmom-cli BeatTracker single <sample.wav>` даёт
  побитово те же строки, что `tests/data/detections/sample.beat_detector.txt`.

## 5. Фазы

Порядок — по зависимостям и по «кто первым даёт проверяемый результат».
Оценки — относительные (S/M/L), не в часах: оценка кода ≈ 1.2–1.5× строк Python
(за счёт типов и тестов), тесты переносятся почти 1:1.

| фаза | содержание | объём | результат |
|---|---|---|---|
| 0 | каркас, devShell, сборка Triton (core+backend), конвертер ONNX, контракт инференса, дампы эталона | L | `cargo test` зелёный + смоук-инференс через собранный `libtritonserver.so` + ONNX-репозиторий |
| 1 | audio-ядро (signal→stft→spectrogram→filterbank→diff) | L | паритет с `tests/test_audio_*.py` |
| 2 | пре-процессинг NN-входов + Triton in-process (ONNX) | M | входы сетей = дампы эталона; активации из ONNX = `.npz` |
| 3 | `ml/hmm` + DBN/CRF биения, пик-пикинг онсетов, темп | M | `BeatTracker`/`DBNBeatTracker`/`OnsetDetector`/`TempoDetector` = `.txt` |
| 4 | downbeats, notes (ADSR), key, chords (CRF), TCN, GMM-паттерны | L | остальные 12 программ CLI = `.txt` |
| 5 | io/CLI/MIDI (25 программ, `single/batch/online`) | M | `tests/test_bin.py` (105 тестов) перенесён |
| 6 | производительность + публикация (`madmom-eval` отложен, вне первого релиза) | S | бенчмарки + документированные ограничения |

---

### Фаза 0 — каркас и инструменты

### Фаза 0 — статус (2026-11-15, авторская среда)

**Сделано и проверено здесь:**

- каркас workspace: `Cargo.toml` (8 крейтов), `rustfmt.toml`, обновлённый `.gitignore`;
- `flake.nix` — devShell (rust, python+scipy/cython/pytest, cmake, gcc, libclang, ffmpeg),
  проверен `nix-instantiate --parse`; **не вычислялся** (нет сети для nixpkgs);
- `tools/build_reference.sh`, `tools/build_triton_core.sh`, `tools/build_onnx_backend.sh`,
  `tools/triton_env.sh`, `madmom.toml` — проверены `bash -n` / разбором TOML;
- `crates/madmom-io/src/npy.rs` — читатель `.npy`/`.npz` без внешних зависимостей:
  **собран и прогнан здесь** `rustc --test` → 6/6 тестов на реальных фикстурах
  (281 значение, `fps=100`, 2-D активации, object-`fps` в `beats_tcn_tempo.npz`,
  отсутствующий член, паддинг заголовка);
- `crates/madmom-testkit/src/lib.rs` — допуски из §4.2 в виде констант, загрузка фикстур,
  `assert_array_close{,_labeled}`, `assert_detections_close`;
- `crates/madmom-ml/src/inference.rs` — трейт `ActivationModel`, `InputShape`,
  `ActivationSource::{GoldenFiles, TritonLocal}` (`triton-live` опционален; без фичи сборка
  не требует `libtritonserver.so`), `TritonLocal` возвращает `NotImplemented`, а не паникует;
- `tools/dump_reference.py`, `tools/dump_preproc.py` — постадийные и посетевые дампы
  (цепочка берётся из самого madmom, поэтому не может разойтись с эталоном);
- `tools/onnx_spec.py` + `tools/convert_to_onnx.py` + `tools/tests/test_convert.py`:
  загрузчик pickle (оба шима NumPy 2), инвентарь слоёв, спецификация репозитория (§2.1),
  генерация `config.pbtxt`; тесты: **5 passed, 1 skipped** (пропуск — построение графов,
  нужен пакет `onnx`);
- факты по моделям, полученные инструментами (не из головы): 92 файла, семейства
  `blstm=24 bgru=12 brnn=17 cnn=10 lstm=8 rnn=8 tcn=8 crf=2 gmm=2 feedforward=1`;
  `beats_lstm_1` = 3 слоя LSTM по 25 скрытых, веса на гейтах `(162,25)`;
  `patterns/*` — это **dict**, внутри которого GMM.

**Собрано здесь после починки прав (Task 0.3/0.5):** эталон madmom собран in place
(`tools/build_madmom_ext.py`), `import madmom` работает, золотые дампы стадий сняты
(`tests/golden/`, 8 файлов, 5 МБ) и правило нарезки кадров проверено независимой
реализацией — **побитово**.

**Находки при сборке эталона (обе — ловушки, стоившие прогонов):**

- мадамовский `setup.py` объявляет `setup_requires=['pytest-runner']`, setuptools идёт
  за яйцом через `pip`, а в nix-окружении пипа нет → сборка падает **до** компиляции.
  Обходится без правки сабмодуля: `tools/build_madmom_ext.py` глушит два метода
  `fetch_build_egg*` и запускает `setup.py` как есть (требование обслуживало только
  удалённую команду `setup.py test`);
- `import madmom` читает собственную версию через `importlib.metadata` → без
  метаданных `PackageNotFoundError`. Поэтому тот же скрипт дополнительно выполняет
  `egg_info`; плюс madmom в `__init__` импортирует `mido` (в env добавлен).
  `dump_*.py` получили `--madmom-path`: madmom не установлен, а подключается на месте.

**Правило нарезки кадров (проверено побитово, критерий Фазы 1):**

```
num_frames = ceil(num_samples / hop_size)          # hop = sample_rate/fps = 441
frame_i    = signal[i*hop - frame_size//2 : i*hop + frame_size//2]   # центрированные!
вне сигнала — нули (signal_frame(pad=0))
```

Пересчёт этого правила в numpy (не через madmom) воспроизводит `sample.frames.npy`
**точно** (281, 2048, int16). Две ловушки: кадры **центрированы** на `i*hop` (левый
выравниватель сдвинул бы все стадии на 1024 сэмпла), и число кадров — `ceil(n/hop)`
(хвост дополняется, а не отбрасывается; `1 + (n - frame_size)//hop` даёт 276 — неверно).

Ступени NN-входа: `diff` = 90 = 45 (+1-я разность) + 45 (−1-я), `features` —
уплощение `281 x 90` в вектор 25290 — это и есть вход сети.

**Не проверено здесь:** вычисление `flake.nix` (нет сети для nixpkgs — только
`nix-instantiate --parse`), сборка `libtritonserver.so` + ONNX Runtime backend,
смоук-инференс, ONNX-графы (`onnx.checker`), числовые проверки графов.

**Гейты, прогнанные после исправления прав (cargo 1.97.0, офлайн из кэша реестра):**

```
cargo build --offline --workspace          # OK
MADMOM_FIXTURES=$PWD/third_party/madmom/tests/data cargo test --offline --workspace   # 6 passed
cargo clippy --offline --workspace --all-targets -- -D warnings   # чисто
cargo fmt --all --check                    # чисто
```

Найденное и исправленное на этом гейте (коммиты `f4ea8f0`, `b108d38`, `11bf680`):

- `optional = true` недопустим в `[workspace.dependencies]` — признак принадлежит
  крейту-члену (`madmom-ml`), а не workspace;
- `ndarray = "0.16"` неразрешим (в кэше/релизах — `0.15.6`/`0.17.2`), зафиксировано `0.17`;
- **зависимость `tritonserver-rs` отложена до Task 2.3**: cargo разрешает
  зависимости path-крейта даже когда он объявлен `optional` и не используется, а
  `cuda-driver-sys` из его набора отсутствует в кэше → любой офлайн-билд workspace
  падал. Фича `triton-live` осталась (без `dep:`), маппинг возвращается в Task 2.3.
  В `Cargo.toml` оставлен комментарий с точной инструкцией, чтобы это не потерялось;
- `Cargo.lock` закоммичен (сборка с бинарями — лок должен быть в репозитории).

Появившийся `flake.lock` означает, что `nix develop` на хосте вычисляется успешно.

**Состояние конвертера и приёмки ONNX (Task 0.8, прогнано здесь):**

- `tools/convert_to_onnx.py --validate --report` по всем 92 моделям: `converted=1`
  (`chroma_dnn`, feedforward, `onnx.checker` проходит), `unsupported=91` — это честный
  список задач Фазы 2, а не пропуск: `rnn`/`brnn`/`gru`/`bgru` (49) ждут билдеров
  рекуррентных узлов, `lstm`/`blstm` (32) — примитивной рекуррентности с пипхолами,
  `cnn` (10)/`tcn` (8)/`crf` (2)/`gmm` (2) — своих билдеров (Фазы 4–5);
- `tools/validate_onnx.py` — численная сверка ONNX против эталона: вход берётся из
  `dump_preproc.py` (`tests/golden/preproc/*.npy` — то, что получает сеть), выход — из
  `tests/golden/activations/*.npy`, допуски §4.2 (|Δ| ≤ 1e-4, косинус ≥ 0.999), код возврата
  ненулевой при расхождении. Инструмент готов, применить его сейчас нельзя: модели-цели
  (LSTM/BLSTM) пока не конвертируются;
- `tools/tests/test_convert.py` с пакетом `onnx` (в nixpkgs есть, 1.21.0): **6 passed** —
  `onnx.checker` проходит на построенном графе `chroma_dnn`;
- измеренная ширина входа сетей: offline-цепочка биений — 90 (2×45), онлайн
  (`RNNBeatProcessor(online=True)`, `beats_lstm`) — **162** при 12 бэндах 30…17000 Гц,
  `log10`, `SpectrogramDifferenceProcessor(diff_ratio=0.5)`. Разложение 162 по стадиям —
  работа Фазы 1 (промежуточные объекты madmom ленивые, форму нужно измерять по стадиям).


#### Task 0.0: Сабмодули (иначе нечего собирать)

**Files:** Modify — ничего; это предполётная проверка окружения.

- [ ] Шаг 1: `git submodule update --init --recursive` — выкачиваются `third_party/madmom`
      (исходники + фикстуры), вложенный `third_party/madmom/madmom/models` (92 `.pkl`, 35 МБ)
      и `third_party/tritonserver-rs` вместе с его вложенным `core` (r24.08).
- [ ] Шаг 2: проверить: `ls third_party/madmom/tests/data/activations | wc -l` → 24;
      `find third_party/madmom/madmom/models -name '*.pkl' | wc -l` → 92;
      `ls third_party/tritonserver-rs/core/include/triton/core/tritonserver.h` → есть.
- [ ] Шаг 3: замечание по состоянию репозитория (не править без решения владельца):
      в корневом `.gitmodules` остался `third_party/madmom_models`, но каталога нет —
      модели фактически приходят из вложенного сабмодуля `madmom/models`. Либо удалить
      лишнюю запись, либо выкачать её тоже (тогда будет два источника одних и тех же файлов).

#### Task 0.1: Workspace и стиль

**Files:** Create `Cargo.toml`, `crates/*/Cargo.toml`, `rustfmt.toml`, `.cargo/config.toml`.

**Interfaces (Produces):** workspace с членами `madmom-core`, `madmom-audio`,
`madmom-ml`, `madmom-features`, `madmom-io`, `madmom-eval`, `madmom`, `madmom-cli`;
edition 2021+, `resolver = "2"`, общие `[workspace.dependencies]`.

- [ ] Шаг 1: создать каркас крейтов с пустыми `lib.rs` (только crate-level doc).
- [ ] Шаг 2: `cargo build --workspace` → успех; `cargo clippy -- -D warnings` → чисто.
- [ ] Шаг 3: коммит `chore: scaffold workspace`.

#### Task 0.2: `flake.nix` devShell

**Files:** Create `flake.nix`.

**Interfaces (Produces):** devShell с `rustc`/`cargo`/`clippy`/`rustfmt`,
`python3.withPackages [numpy scipy cython pytest packaging]`, `ffmpeg`, `git`.

- [ ] Шаг 1: `nix develop` → `cargo --version`, `python3 -c "import numpy, scipy, Cython"`, `ffmpeg -version`.
- [ ] Шаг 2: `nix develop --command cargo test` → зелено.
- [ ] Шаг 3: коммит `chore: add nix dev shell`.

#### Task 0.3: Эталонная сборка madmom и baseline тестов

**Files:** Create `tools/build_reference.sh`.

**Interfaces (Produces):** собранный `third_party/madmom` (`build_ext --inplace`)
и зафиксированный список падающих тестов в `tools/reference-baseline.txt`.

- [ ] Шаг 1: скрипт: `nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython ps.pytest ps.packaging])' --run 'cd third_party/madmom && python3 setup.py build_ext --inplace && python3 -m pytest tests -q -p no:cacheprovider' > tools/reference-baseline.txt`.
- [ ] Шаг 2: запустить, записать фактический результат (ожидание: часть тестов
      падает из-за отсутствия opencv/pyfftw/pyaudio — это фиксируется, а не «исправляется»).
- [ ] Шаг 3: коммит `chore: reference build script + baseline`.

> Известный риск: `setup.py` требует `Cython` и компилирует `ml/nn/layers.py`
> через Cython; при провале — включаем только `hmm`/`comb_filters`/`beats_crf`
> и фиксируем, какие тесты из-за этого не проходят.

#### Task 0.4: Контракт инференса и golden-режим (модели не трогаем)

Модели не портируем в Rust: они уезжают в ONNX (конвертер — наш, Task 0.8, §2.1),
инференс — `tritonserver-rs` in-process (сабмодуль в `third_party/`). Чтобы Фазы 3–5
не зависели от готовности ONNX-репозитория (`third_party/madmom-models-onnx`, §2.1):
активации берутся из двух источников за одним трейтом (детали — Фаза 2).

**Files:** Create `crates/madmom-ml/src/inference.rs` (трейт),
`tools/dump_preproc.py`; Test `crates/madmom-ml/tests/inference.rs`.

**Interfaces (Produces):** `ActivationModel`, `ActivationSource::GoldenFiles`
(см. §3.3 и Task 2.1) — здесь заводится только трейт и golden-путь.

- [ ] Шаг 1: трейт `ActivationModel` + `ActivationSource::GoldenFiles`.
- [ ] Шаг 2 (тест): golden-источник отдаёт активации для `RNNBeatProcessor`-цепочки;
      downstream-стадии (HMM, пик-пикинг) работают без сети.
- [ ] Шаг 3: отсутствие `MADMOM_MODEL_REPO`/репозитория даёт понятную ошибку только
      при выборе `ActivationSource::TritonLocal`, а не при старте и не в golden-режиме.
- [ ] Шаг 4: коммит `feat(ml): activation source contract (golden + triton stub)`.

#### Task 0.5: Дампы промежуточных стадий эталона

**Files:** Create `tools/dump_reference.py`; (выход) `tests/golden/`.

**Interfaces (Produces):** `tests/golden/<pipeline>/<stage>.npy` для
`sample.wav`, `sample2.wav`, `stereo_sample.wav`; стадии: `signal`, `frames_2048`,
`stft_2048`, `spec_2048`, `filt_6_30_17000`, `logspec`, `diff_0.5_pos`,
`activations_blstm`, `activations_lstm`, `path_dbn`, `beats`.

- [ ] Шаг 1: скрипт импортирует madmom, проходит цепочку постадийно
      (сохраняя массивы через `np.save`), для биений — DBN-путь и финальные детекции.
- [ ] Шаг 2: запустить; проверить, что `beats` совпадают с
      `third_party/madmom/tests/data/detections/sample.beat_detector.txt`.
- [ ] Шаг 3: коммит `test: reference stage dumps for sample.wav`.

#### Task 0.6: Тестовая обвязка (testkit + npz)

**Files:** Create `crates/madmom-io/src/npy.rs` (npy/npz), `crates/madmom-testkit/`
(dev-only крейт: `assert_array_close!`, загрузка фикстур); Test `crates/madmom-io/tests/npz.rs`.
Плюс `tests/fixtures -> ../../third_party/madmom/tests/data` (симлинк, чтобы тесты
ссылались на данные по короткому пути).

**Interfaces (Produces):**
```rust
pub fn read_npy(path: &Path) -> Result<(DType, Vec<usize>, Vec<u8>)>;
pub fn read_npz_member(path: &Path, key: &str) -> Result<(DType, Vec<usize>, Vec<u8>)>;
pub fn assert_array_close(a: &ArrayD<f32>, b: &ArrayD<f32>, atol: f32, rtol: f32);
```

- [ ] Шаг 1: тест: чтение `tests/fixtures/activations/sample.beats_blstm.npz`,
      форма `activations=(281,) float32`, скаляр `fps=100` (проверено на эталоне:
      у `sample.deep_chroma.npz` — `(29,12) float32, fps=10`).
- [ ] Шаг 2: тест на файл с 0-d `fps` (object-array) — не падать, вернуть `fps=None`.
- [ ] Шаг 3: коммит `feat(io): npy/npz readers + testkit assertions`.

#### Task 0.7: Triton runtime из исходников: `libtritonserver.so` + ONNX Runtime backend

Решение владельца: **собираем `core` сами** (не контейнер). Исходники уже на месте —
сабмодуль `third_party/tritonserver-rs/core` (ветка r24.08, C-API 1.34).

**Files:** Create `tools/build_triton_core.sh`, `tools/build_onnx_backend.sh`,
`tools/triton_env.sh`; (выход, gitignored) `third_party/triton-runtime/{lib,backends}`.

- [ ] Шаг 1: `libtritonserver.so` (из `core/CMakeLists.txt`: по умолчанию собирается
      только заголовок+заглушка, полная библиотека — только с `TRITON_CORE_HEADERS_ONLY=OFF`):
      ```bash
      cmake -S third_party/tritonserver-rs/core -B build/triton-core -DCMAKE_BUILD_TYPE=Release \
        -DTRITON_CORE_HEADERS_ONLY=OFF -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_MALI_GPU=OFF \
        -DTRITON_ENABLE_METRICS_GPU=OFF -DTRITON_ENABLE_METRICS=OFF -DTRITON_ENABLE_STATS=OFF \
        -DTRITON_ENABLE_TRACING=OFF -DTRITON_ENABLE_NVTX=OFF -DTRITON_ENABLE_ENSEMBLE=OFF \
        -DTRITON_ENABLE_GCS=OFF -DTRITON_ENABLE_S3=OFF -DTRITON_ENABLE_AZURE_STORAGE=OFF \
        -DTRITON_COMMON_REPO_TAG=r24.08 -DTRITON_THIRD_PARTY_REPO_TAG=r24.08 \
        -DCMAKE_INSTALL_PREFIX=$PWD/third_party/triton-runtime
      ```
      Теги `*_REPO_TAG` **обязаны** быть `r24.08` (по умолчанию `main` — несовместимо с
      ядром r24.08). Метрики/статистика выключены: `METRICS=ON` требует `STATS=ON`, а
      `METRICS_GPU=ON` — `GPU=ON` (иначе CMake падает с FATAL_ERROR).
- [ ] Шаг 2: учесть, что `core` тянет и **собирает** третьи стороны (gRPC, protobuf,
      absl, re2, c-ares, prometheus-cpp, googletest) из `triton-inference-server/third_party`
      через FetchContent/ExternalProject → нужна сеть при конфигурации и ~гигабайты
      сборки. Герметичная nix-деривация для этого — отдельная задача (см. §9, вопрос 2).
- [x] Шаг 3: **ONNX Runtime backend** — сабмодуль `third_party/onnxruntime_backend`,
      **выровнен по ядру: ветка `r24.08`** (коммит `ab34b48`, 2024-08-15) — сделано
      владельцем. Проверено: требования этой ветки — CMake ≥ 3.17 и C++17, т.е. совпадают
      с `core` r24.08. Связка версий согласована: `core` r24.08 (C-API 1.34) +
      backend r24.08 + bindings `tritonserver-rs` 0.4.1 (минимальный контейнер 24.08).
- [ ] Шаг 3a: **ONNX Runtime вендорим** (решение владельца; системный ORT не используем).
      Проверенный по `CMakeLists.txt` backend'а набор возможностей:
      (а) внешний ORT — `-DTRITON_ONNXRUNTIME_INCLUDE_PATHS=…/include`
      `-DTRITON_ONNXRUNTIME_LIB_PATHS=…/lib` (внутри — `find_library(onnxruntime … REQUIRED)`);
      (б) `-DTRITON_BUILD_ONNXRUNTIME_VERSION=<ver>` + `-DTRITON_BUILD_CONTAINER=<image>` —
      сборка ORT **в контейнере** (docker недоступен в этом окружении);
      (в) `-DTRITON_ONNXRUNTIME_PACKAGE_URL=<zip>` — скачивание готового пакета ORT
      (`cmake/download_onnxruntime.cmake`: ждёт zip с `VERSION_NUMBER`, `include/`, `lib/`,
      сверяет версию с `TRITON_BUILD_ONNXRUNTIME_VERSION`).
      Выбираем **(а) + наш пиннинг**: официальный ORT-релиз скачиваем один раз (URL + sha256
      фиксируем в `tools/build_onnx_backend.sh`), распаковываем в
      `third_party/triton-runtime/onnxruntime/` и передаём пути.
      **Версия ORT:** в самом репозитории бэкенда она не зафиксирована (её подаёт CI через
      `TRITON_BUILD_ONNXRUNTIME_VERSION`), поэтому берём **эпохально совпадающую с r24.08
      (август 2024) — `1.18.1`** как стартовую, sha256 пишем в скрипт; привязка по ABI —
      вендоренная `libonnxruntime.so.1` рядом с backend'ом + rpath/`LD_LIBRARY_PATH`.
      Если понадобится ровно та версия, что в образе `tritonserver:24.08-py3`, её можно
      прочитать из образа и обновить пиннинг (это одна строка в скрипте).
      Флаги: `-DTRITON_ENABLE_GPU=OFF` (иначе `TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON`
      падает с FATAL_ERROR, т.к. требует GPU), `-DTRITON_ENABLE_STATS=OFF`.
      Результат — `libtriton_onnxruntime.so` (+ вендоренные ORT-либы) в
      `third_party/triton-runtime/backends/`.
- [ ] Шаг 4: `tools/triton_env.sh` — экспорт `MADMOM_MODEL_REPO`,
      `MADMOM_TRITON_BACKENDS=$PWD/third_party/triton-runtime/backends`,
      `LD_LIBRARY_PATH=$PWD/third_party/triton-runtime/lib` (нужен и для линковки
      `tritonserver-rs` — `LIBRARY_PATH` при сборке, `LD_LIBRARY_PATH` при запуске).
- [ ] Шаг 5 (смоук-тест, до всей Фазы 2): крошечная ONNX-модель (`add`) + `config.pbtxt`
      с `KIND_CPU` → пример `examples/smoke_triton.rs`: сервер стартует, `is_ready()`
      истинно, один инференс даёт ожидаемые числа. Только после этого идёт Task 2.3.
- [ ] Шаг 6: коммит `build(triton): build libtritonserver + onnxruntime backend from source`.

#### Task 0.8: Конвертер моделей → ONNX (пишется до Фазы 2)

**Files:** Create `tools/convert_to_onnx.py` (исполняемая спецификация §2.1/§2.2),
`tools/onnx_spec.py` (общие хелперы: упаковка гейтов, свёртки, TCN);
Test `tools/tests/test_convert.py`.

- [ ] Шаг 1: загрузчик pickle по D1 (с шимами NumPy 2), выгрузка весов в numpy.
- [ ] Шаг 2: сборка графов по таблице «слой madmom → ONNX» (§2.1) + генерация
      `config.pbtxt` (CPU-only) для каждой модели; имена каталогов — из §2.1.
- [ ] Шаг 3 (тест): для каждой сконвертированной модели — прогон через `onnxruntime`
      на эталонных признаках (`tests/golden/preproc/*.npy`) и сравнение с эталонным
      выходом сети (`tests/golden/activations/<model>.npy`), допуск `1e-4`. Тест
      падает с именем модели, а не «что-то не сошлось».
- [ ] Шаг 4: `--report` — сколько моделей сконвертировано, формы входов/выходов,
      sha256 файлов (пригодится в Task 2.4 как ожидаемый список).
- [ ] Шаг 5: коммит `feat(tools): madmom pickle -> onnx converter`.

---

### Фаза 1 — audio-ядро

Принцип: каждая задача = один модуль Python, тесты переносятся из
`tests/test_audio_*.py` (числа уже известны) + сверка с `tests/golden/*.npy`.

#### Task 1.1: `signal`: Signal/framing/функции

**Files:** Create `crates/madmom-audio/src/signal.rs`;
Test `crates/madmom-audio/tests/signal.rs` (порт `tests/test_audio_signal.py`, 87 тестов).

**Interfaces:**
- Consumes: `madmom-io::audio::{load_audio_file, load_wave_file}` (заглушки на этой задаче).
- Produces: `Signal`, `FramedSignal`, `SignalProcessor`, `FramedSignalProcessor`,
  `signal_frame()`, `smooth()`, `adjust_gain()`, `attenuate()`, `normalize()`,
  `remix()`, `rescale()`, `trim()`, `energy()`, `root_mean_square()`,
  `sound_pressure_level()`, `write_wave_file()`.

- [x] Шаг 1 (тест): `FramedSignal` на 9-сэмпловом сигнале: `num_frames` для
      `end='normal'`/`'extend'`, `origin='online'`/`'offline'`/`'stream'`,
      `frame_rate`, `overlap_factor`, срез `[2:5]` даёт новый `FramedSignal`
      со сдвинутым origin.
- [x] Шаг 2 (тест): `signal_frame` на границах: `start<0`, `stop>len`,
      `pad='repeat'` (повтор первого/последнего сэмпла) и `pad=0`.
- [x] Шаг 3 (реализация): `signal_frame` — точный перенос формулы
      `start = index*hop - frame_size/2 - origin`.
- [x] Шаг 4 (тест): `smooth` (kernel int → окно Хэмминга, `np.convolve 'same'`;
      2-D → `convolve2d 'same'`), `normalize` (int16 → /32767), `remix`
      (моно=среднее, `channel=N` — выбор канала), `rescale`, `trim`, `energy`,
      `rms`, `spl`.
- [x] Шаг 5: вместо ручного переноса 41 doctest-примера — генератор
      `tools/gen_audio_vectors.py`, который снимает числа у самого madmom и пишет
      `crates/madmom-audio/tests/signal_vectors.rs` (9 тестов: smooth 1-D/2-D с чётным и
      нечётным ядром, gain/attenuate/normalize/rescale, remix, trim, energy/RMS/SPL,
      `signal_frame` с тремя origin, счёт кадров normal/extend). Примеры, требующие
      загрузки файлов (`Signal(path)`, `resample`, `write`), закрываются в Task 1.2.
- [x] Шаг 6: коммит `feat(audio): signal, framing and signal functions`.

**Статус: выполнено (2026-11-15).** Реализовано в `crates/madmom-audio/src/signal.rs`
(≈800 строк с документацией): константы, `DType` с правилами приведения, `Signal`,
`signal_frame`, `FramedSignal`, `smooth`/`hamming`/`convolve_same`, `adjust_gain`,
`attenuate`, `normalize`, `remix`, `rescale`, `trim`, `energy`, `root_mean_square`,
`sound_pressure_level`, `Origin`, `EndOfSignal`.

Проверено здесь (cargo 1.97, офлайн):

```
cargo test -p madmom-audio   → 7 unit + 1 golden + 9 векторов эталона, все зелёные
cargo test -p madmom-io      → 7 (добавлен int16 → f32: без него дампы сигнала не читались)
cargo clippy --workspace --all-targets -- -D warnings → чисто; cargo fmt --check → чисто
```

Ключевое: `tests/signal_golden.rs` сравнивает нарезку реального файла с дампом эталона
**побитово** (`max_abs_diff == 0`, 281 × 2048, int16), а `signal_vectors.rs` — числа самого
madmom для скалярных функций. По ходу всплыл дефект Phase 0: `npy.as_f32()` не умел
целочисленные типы и отказывался читать `sample.signal.npy` — добавлены I16/U16/U32/U64
с тестом.


#### Task 1.2: `io::audio` (WAV + ffmpeg)

**Files:** Create `crates/madmom-io/src/audio.rs`; Test `crates/madmom-io/tests/audio.rs`
(порт `tests/test_io_audio.py`, 27 тестов).

- [x] Шаг 1 (тест): `load_wave_file` для `sample.wav`, `sample_22050.wav`,
      `stereo_sample.wav` (int16, num_channels, sample_rate); ошибка
      `LoadAudioFileError` для `foo_bar.wav`; `write_wave_file` round-trip.
- [ ] Шаг 2 (тест): `load_audio_file` dispatch: `.wav` без ресемплинга → свой
      ридер; `.m4a`/`.flac`/нужен ресемплинг → ffmpeg; `start`/`stop`/`channel`/`gain`/`norm`.
- [ ] Шаг 3 (тест): ReplayGain (`stereo_sample_rg.flac`, `stereo_chirp_rg.flac`) —
      ожидаемые значения из `tests/test_io_audio.py`.
- [ ] Шаг 4 (реализация): `_ffmpeg_call` — аргументы в том же порядке
      (`-v quiet -y`, `-ss`, `-f`, `-t`, `-ac`, `-ar`, `-af pan=mono|c0=cN`).
- [ ] Шаг 5: коммит `feat(io): wav + ffmpeg audio loading`.

**Статус: половина сделана (2026-11-15).** `crates/madmom-io/src/audio.rs`: собственный
RIFF/WAVE-ридер с конвенциями scipy (`wavfile.read`), потому что популярные крейты дают
другие числа: 8-бит — **беззнаковый**, 24-бит — как `int32` со знаковым расширением и
**без** масштабирования, `WAVE_FORMAT_EXTENSIBLE` — по под-формату GUID (тег лежит по
смещению 24, не 18: после 18 байт `WAVEFORMATEX` идут valid-bits, channel-mask и только
потом сам GUID). Плюс обрезка `start`/`stop` с округлением как у эталона
(`int(s*rate)`, `stop = min(len, int(stop*rate))`) и отказ при несовпадении
sample rate/dtype вместо тихой пересборки.

Проверено здесь побитово: `cargo test -p madmom-io --test audio_golden` прогоняет 9 кейсов
дампов madmom (5 файлов + 4 варианта обрезки, включая стерео) и требует совпадения
формата, формы, частоты и **всех** сэмплов — совпало всё. Плюс 7 unit-тестов на парсер
(8/16/24-бит, расширяемый заголовок, стерео, отказ на чужих форматах). Всего в крейте
14 unit + 1 golden.

Осталось по задаче (вторая половина): путь ffmpeg — `ffmpeg_fmt`/`ffmpeg_call`
(**порядок аргументов — часть контракта**: `-ss` до `-i`, `-af volume=replaygain=...` после,
`-f fmt`, `-t`, `-ac`, `pan=mono|c0=cN`, `-ar`), `decode_to_memory`/`decode_to_pipe`,
`get_file_info`, ReplayGain-фикстуры (`stereo_sample_rg.flac`, `stereo_chirp_rg.flac`) и
не-потоковый `.m4a`. ffmpeg в песочнице есть (`/run/current-system/sw/bin/ffmpeg`), так что
эта половина тоже проверяема здесь: эталон снимается у самого madmom при том же бинаре.

#### Task 1.3: `stft`

**Files:** Create `crates/madmom-audio/src/stft.rs`; Test `…/tests/stft.rs`
(порт `tests/test_audio_stft.py`, 27 тестов).

**Interfaces (Produces):** `stft()`, `fft_frequencies()`, `phase()`,
`local_group_delay()`, `ShortTimeFourierTransform` (+ `Processor`),
`Phase`, `LocalGroupDelay`, `ShortTimeFourierTransformProcessor`.

- [x] Шаг 1 (тест): doctest-значения `fft_frequencies(4, 44100)`;
      `stft` без окна/с окном Ханна, `circular_shift=True` (свап половин
      с дополнением нулями при `fft_size > frame_size`), `include_nyquist`.
- [x] Шаг 2 (реализация): масштабирование окна `window / iinfo(dtype).max`
      при целочисленном сигнале (D5) — иначе паритет невозможен.
- [x] Шаг 3 (тест): сверка `stft` с `tests/golden/stft_2048.npy` (`≤1e-5`).
- [x] Шаг 4: коммит `feat(audio): short-time Fourier transform`.

**Статус: выполнено (2026-11-15).** `crates/madmom-audio/src/stft.rs`: `stft()`,
`ShortTimeFourierTransform`, `ShortTimeFourierTransformProcessor`, `fft_frequencies`,
`phase`, окна `hanning`/`hamming` (формулы NumPy: `0.5-0.5cos`, `0.54-0.46cos`), `Window`
(функция/массив/нет) и правило масштабирования окна.

Числа (прогнано здесь, `cargo test -p madmom-audio`):

- сверка с дампом `tests/golden/sample.stft.npy` (281 × 1024, complex64):
  **max |Δ| = 2.910e-11** при допуске §4.2 = 1e-5, то есть на шесть порядков внутри;
- тест-ловушка: то же окно **без** деления на 32767 даёт `max |Δ| = 4.644e6` — то есть
  забытое масштабирование видно, а не «правдоподобно» уезжает;
- 3 теста на векторах, снятых с madmom, закрывают то, чего нет в дампе: `fft_frequencies`
  (5,10), (4,44100), (1024,44100); окна 1/2/3/4/8/2048; случаи `include_nyquist`,
  `circular_shift`, `fft_size=16 > кадра`, `fft_size=4 < кадра`, кадры float без масштабирования.

Важные детали, зафиксированные в коде: madmom считает **в float64** (int16 × float64-окно)
и только результат кладёт в `complex64`, поэтому FFT здесь тоже в `f64`; `num_bins = fft_size/2`
(Найквист выбрасывается, 2048 → 1024 бина); при `circular_shift` половины кадра меняются
до FFT, а при `fft_size > frame_size` нули вставляются **между** половинами; при
`fft_size < frame_size` кадр **обрезается** (`fftpack.fft`-семантика) — это нашёл
вектор-тест, до него реализация падала на выходе за границы.

Попутно: `madmom-io::npy` научился читать `complex64/128` (`as_complex32`); ошибка в
`size_in_bytes` (complex64 как 4 байта вместо 8) была поймана тем же тестом.

#### Task 1.4: `filters` (фильтрбанки)

**Files:** Create `crates/madmom-audio/src/filters.rs`;
Test `…/tests/filters.rs` (порт `tests/test_audio_filters.py`, 97 тестов).

**Interfaces (Produces):** `Filter`, `TriangularFilter`, `RectangularFilter`,
`Filterbank`, `MelFilterbank`, `BarkFilterbank`, `LogarithmicFilterbank`,
`RectangularFilterbank`, `SimpleChromaFilterbank`, `HarmonicFilterbank`,
`PitchClassProfileFilterbank`, `HarmonicPitchClassProfileFilterbank`,
`SemitoneBandpassFilterbank`, функции `hz2mel/mel2hz/mel_frequencies/hz2bark/
bark2hz/bark_frequencies/bark_double_frequencies/log_frequencies/
semitone_frequencies/hz2midi/midi2hz/hz2erb/erb2hz/frequencies2bins/bins2frequencies`.

- [x] Шаг 1 (тест): все скалярные преобразования частот (doctest-числа).
- [x] Шаг 2 (тест): матрицы `MelFilterbank(40 bands, 20..17000)`,
      `LogarithmicFilterbank(12/octave, num_bands=6/12/24, 30..17000, norm=True)`;
      `frequencies2bins(unique=True)` — «ближайший бин» с округлением
      (`indices -= freqs - left < right - freqs`).
- [ ] Шаг 3: `SemitoneBandpassFilterbank` (`ellip`+`filtfilt`) — реализуется
      последней в задаче, допускается отдельный коммит и `#[cfg]`-флаг, если
      эллиптический фильтр ещё не готов (тест не удалять).
- [ ] Шаг 4: коммит `feat(audio): filterbanks`.

**Статус: сделана первая часть (2026-11-15).** `crates/madmom-audio/src/filters.rs`:
преобразования частот (`hz2mel`/`mel2hz`, `mel_frequencies`, `hz2midi`/`midi2hz`, `hz2erb`/
`erb2hz`, `log_frequencies`, `semitone_frequencies`, таблицы Bark), `frequencies2bins` /
`bins2frequencies`, `Filter`/`triangular_filter`/`rectangular_filter`, `triangular_band_bins`/
`rectangular_band_bins`, `Filterbank` (+`from_filters`,`apply`), `mel_filterbank`,
`logarithmic_filterbank`, `bark_filterbank` (нормальный/двойной шаг).

Проверено против 20 дампов madmom (`tools/dump_filters.py` → `tests/golden/filters/`):
**max |Δ| = 1.490e-08** при границе 1e-6 — расхождение только в порядке суммирования
float32 при нормировке (NumPy суммирует попарно). Плюс 8 unit-тестов.

Три отличия Bark-фильтрбанка, которые нашлись при сверке (и подтверждены дампом):
он строится из **прямоугольных** неперекрывающихся фильтров (а не треугольных), флаг
`unique_filters` у него **инвертирован** (`unique_bins = not unique_filters`), а таблица
двойного шага начинается `20, 50, 100…` — в моей первой версии было `60`, и это поймал
дамп, а не глаз. Кроме того `hz2bark`/`bark2hz` в эталоне **падают** с `NotImplementedError`
(«produces negative values») — в порте они тоже возвращают ошибку, а не выдуманные числа.

**Осталось по задаче:** `SemitoneBandpassFilterbank` (нужны `scipy.signal.ellip` +
`filtfilt`), хрома-фильтрбанки (`SimpleChromaFilterbank`, `PitchClassProfileFilterbank`,
`HarmonicFilterbank`, `HarmonicPitchClassProfileFilterbank`) и `RectangularFilterbank`.
Это отдельный шаг: хрома нужна Фазе 5, полосовые фильтры — нотным/хрома-цепочкам.

#### Task 1.5: `spectrogram` (Spectrogram/Filtered/Log/Diff/MultiBand)

**Files:** Create `crates/madmom-audio/src/spectrogram.rs`;
Test `…/tests/spectrogram.rs` (порт `tests/test_audio_spectrogram.py`, 49 тестов).

- [x] Шаг 1 (тест): `Spec` = |STFT|; `FilteredSpectrogram` = `spec · filterbank`;
      `LogarithmicSpectrogram` (`log10`, `mul`, `add`); `SpectrogramDifference`
      (`diff_ratio`, `diff_frames`, `diff_max_bins`, `positive_diffs`, `stack_diffs`);
      `SuperFluxProcessor` (композиция).
- [x] Шаг 2 (тест): сверка с `tests/golden/{filt_6_30_17000,logspec,diff_0.5_pos}.npy`.
- [ ] Шаг 3 (тест): `MultiBandSpectrogram`, `SemitoneBandpassSpectrogram` —
      на известных из тестов madmom значениях.
- [ ] Шаг 4: коммит `feat(audio): spectrograms and differences`.

**Статус: сделана основная часть (2026-11-15).** `crates/madmom-audio/src/spectrogram.rs`:
`spec` (|STFT|), `Spectrogram` (стадии плоским набором: `filter`, `logarithmic`,
`difference`, `filter_with`, `positive_diff`), `diff_frames` (порт `_diff_frames`),
`DifferenceParameters`, `StackDiffs` (нет/горизонтально/вертикально; глубина — отказ),
процессоры `SpectrogramProcessor`, `FilteredSpectrogramProcessor`,
`LogarithmicSpectrogramProcessor`, `LogarithmicFilteredSpectrogramProcessor` (та, что
используют цепочки биений) и `SpectrogramDifferenceProcessor`. В `Filterbank` добавлены
`center_frequencies`/`corner_frequencies`/`fmin`/`fmax` (нужны как частоты бинов после
фильтрации) и они сверены с эталоном.

Сверка постадийно против дампов `sample.*.npy` (граница 1e-5):

```
spec         max |Δ| = 7.629e-6      logspec      max |Δ| = 1.192e-7
filt         max |Δ| = 7.629e-6      diff         max |Δ| = 2.384e-7
logspec(1x)  max |Δ| = 1.192e-7   (комбинированный процессор == раздельные)
```

7.629e-6 — это ровно **один ulp float32** при величинах 64…128: обе реализации FFT
сходятся в float64, а приведение к `complex64` переворачивает последний бит у части
значений. Это и есть причина, по которой допуск для спектрограмм 1e-5, а не 1e-9 —
записано в README дампов. `diff_frames`, вычисленный из `diff_ratio=0.5` и окна
2048/441, совпал с эталоном (=1); первые `diff_frames` строк разности нулевые, как и
должно быть из-за трюка с `+inf`.

**Осталось по задаче:** `MultiBandSpectrogram` (несколько фильтрбанков на одном STFT),
`SemitoneBandpassSpectrogram` (нужен полутоновый фильтрбанк из Task 1.4, то есть
`ellip`+`filtfilt`), `diff_max_bins > 1` (нужна семантика `scipy.ndimage.maximum_filter`)
и глубинное стекирование (`dstack`). Все четыре — явные отказы в коде, а не тихие пропуски.

#### Task 1.6: `chroma`, `cepstrogram`, `hpss`, `comb_filters`

**Files:** Create `crates/madmom-audio/src/{chroma,cepstrogram,hpss,comb_filters}.rs`;
Test `…/tests/{chroma,cepstrogram,hpss,comb_filters}.rs` (портировать
`tests/test_audio_chroma.py`, `tests/test_audio_comb_filters.py` + doctest-числа).

- [ ] Шаг 1 (тест): `feed_forward_comb_filter`/`feed_backward_comb_filter` —
      doctest-примеры (`[0,0,1,0,0,1,0,0,1]`, tau=3, alpha=0.5 → `[0,0,1,0,0,1.5,0,0,1.5]`
      и `[…,1.75]`).
- [ ] Шаг 2 (тест/реализация): `CombFilterbankProcessor`, `Cepstrogram`/`MFCC`
      (DCT-II, `MFCC_BANDS=30`, 40..15000), `HarmonicPercussiveSourceSeparation`
      (`median_filter`, `HARMONIC_FILTER=(15,1)`, `PERCUSSIVE_FILTER=(1,15)`,
      masking `binary`/`ratio`), `PitchClassProfile`, `CLPChroma`.
- [ ] Шаг 3: коммит `feat(audio): chroma, cepstrogram, hpss, comb filters`.

---

### Фаза 2 — инференс через Triton (вместо NN-движка)

Решение владельца: **NN-слои в Rust не портируются**. `madmom.ml.nn` (1454 строки
слоёв + `activations.py` + Cython-компиляция `layers.py`) заменяется вызовом модели
на `tritonserver` через `tritonserver-rs` (сабмодуль `third_party/tritonserver-rs`,
CPU-only). Обязательства сдвигаются: Rust обязан
воспроизвести **ровно тот вход**, который madmom подавал в первую сеть ансамбля, и
правильно интерпретировать выход. Что именно строится (константы из исходников):

| пайплайн | пре-процессинг (Rust) | вход сети |
|---|---|---|
| биения offline (`BEATS_BLSTM`) | 3 разрешения (frame 1024/2048/4096) × 6 бэндов 30…17000, `norm_filters=True`, `log10(1+x)`, положительные диффы `diff_ratio=0.5`, `hstack` | 2-D `(frames, features)` |
| биения online (`BEATS_LSTM`) | frame 2048, 12 бэндов, те же шаги | 2-D |
| биения TCN (`BEATS_TCN`) | frame 2048, 12 бэндов, `log(x+1e-6)`, паддинг 2 кадра с обеих сторон | 3-D `(frames, freq, channels)` |
| онсеты RNN (`ONSETS_BRNN`) | 3 разрешения (1024/2048/4096) × 6 бэндов, `mul=5, add=1`, `diff_ratio=0.25` | 2-D |
| онсеты CNN (`ONSETS_CNN`) | 3 разрешения (2048/1024/4096) при `fps=100`, `MelFilterbank(80, 27.5…16000)`, `log`, `dstack`, паддинг 7 кадров | 3-D |
| downbeats (`DOWNBEATS_BLSTM`, `DOWNBEATS_BGRU`) | как биения + `MultiBandSpectrogram` (ритмические/гармонические фичи) | 2-D |
| deep chroma (`CHROMA_DNN`) | 1 разрешение (`MultiBandSpectrogram`, выход на `fps=10`) | 2-D |
| key (`KEY_CNN`) | CNN-семантика `PadLayer`/`AverageLayer`, вход 3-D | 3-D |
| notes CNN (`NOTES_CNN`/`_MIREX`) | 3 разрешения + `BatchNorm`, вход 3-D | 3-D |

Постобработка (усреднение ансамбля, `squeeze`, `itemgetter(0)` для multi-task TCN,
пик-пикинг, HMM) остаётся в Rust.

#### Task 2.1: Трейт, контракт и golden-режим

**Files:** Create `crates/madmom-ml/src/inference.rs`; Test `crates/madmom-ml/tests/inference.rs`.

**Interfaces (Produces):**
```rust
pub enum InputShape { Frames(usize), FramesFreq(usize, usize), FramesFreqCh(usize, usize, usize) }
pub trait ActivationModel {
    fn input_shape(&self) -> InputShape;
    fn predict(&mut self, features: &ArrayD<f32>) -> Result<ArrayD<f32>>;
}
pub enum ActivationSource { GoldenFiles(PathBuf), Triton(TritonModel) }   // оба impl ActivationModel
```

- [ ] Шаг 1: трейт + `ActivationSource::GoldenFiles` (читает `.npz`/`.npy`).
- [ ] Шаг 2 (тест): цепочка `RNNBeatProcessor` с golden-источником воспроизводит
      `tests/fixtures/activations/sample.beats_blstm.npz` и далее
      `sample.beat_detector.txt` — без Triton (на этом живут Фазы 3–5).
- [ ] Шаг 3: коммит `feat(ml): activation model trait + golden source`.

#### Task 2.2: Пре-процессинг-«билдеры» под каждую группу моделей

**Files:** Modify `crates/madmom-features/src/{beats,onsets,downbeats,chroma,key,notes}.rs`;
Create `tools/dump_preproc.py`; Test `crates/madmom-features/tests/preproc.rs`.

- [ ] Шаг 1: `tools/dump_preproc.py` — с эталона снять **вход** каждой группы
      (`tests/golden/preproc/<group>.npy`) вместе с выходом.
- [ ] Шаг 2 (тест): `sample.wav` → вход `BEATS_BLSTM` = `golden/preproc/beats_blstm.npy`
      (`≤1e-6`; порядок разрешений и `hstack` фиксированы).
- [ ] Шаг 3 (тест): то же для `ONSETS_BRNN`, `ONSETS_CNN` (3-D + паддинг 7 кадров),
      `BEATS_TCN` (паддинг 2 кадра), `DOWNBEATS_BLSTM`, `CHROMA_DNN`, `KEY_CNN`.
- [ ] Шаг 4: коммит `feat(features): NN input preprocessing builders`.

#### Сборка рантайма (перед Task 2.3)

Собирается в песочнице, из сабмодулей как есть (`tools/build_triton_core.sh` →
`build/triton-core-build.log`, `tools/build_onnx_backend.sh` → `build/ort-backend-build.log`;
общее окружение — `tools/triton_build_env.sh`). Коммит `89ccdd5`.

- **Бэкенд ONNX Runtime — собран**: `third_party/triton-runtime/backends/onnxruntime/libtriton_onnxruntime.so`,
  ONNX Runtime 1.18.1 в `third_party/triton-runtime/onnxruntime/`. Раскладка — та, которую ищет
  Triton (`backends/<имя>/libtriton_<имя>.so`).
- **Ядро** (`libtritonserver.so`) — собирается; `CMAKE_INSTALL_LIBDIR=lib` обязателен, иначе
  ядро не находит собственные third-party (protobuf/re2/abseil): оно ищет их по захардкоженному
  `LIB_DIR`, который равен `lib` везде, кроме CentOS, а GNUInstallDirs ставит в `lib64`.
- protobuf, Abseil, re2, googletest ядро собирает **само** из `third-party@r24.08`; из nixpkgs
  берутся только RapidJSON, numactl и Boost (nixpkgs protobuf 34 против r24.08 не компилируется:
  `field->name()` там `string_view`, а `JsonPrintOptions::always_print_primitive_fields` удалён).
- Ловушки окружения: в песочнице нет `/bin/sh` и `/usr/bin/env` (создаются в том же вызове, что
  и сборка); CMake 4 требует `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` для pybind11 из r24.08;
  `-include cstdint` для `triton-common/error.h`; `-Wno-error=maybe-uninitialized` (GCC 15) в
  `CMAKE_CXX_FLAGS_RELEASE`, чтобы он оказался позже `-Werror` самого проекта.

**Что выяснилось про `tritonserver-rs` 0.4.1 (влияет на шаги ниже):**

- `Options` **не отдаёт** `backend_directory` (`Options.0` — `pub(crate)`, извне не достать),
  а у ядра путь по умолчанию захардкожен: `backend_dir_("/opt/tritonserver/backends")`
  (`core/src/tritonserver.cc:404`). Для дымового теста — симлинк
  `/opt/tritonserver/backends` → `third_party/triton-runtime/backends`; «чистая» альтернатива —
  добавить в биндинги `Options::backend_directory()` (5 строк, правит сабмодуль → **нужно
  решение владельца**).
- `Server::new(options)` — `async`. Достаточно `futures::executor::block_on` (tokio не нужен,
  если не хочется тянуть рантайм; `futures` уже в зависимостях биндингов).
- `build.rs` биндингов гоняет **bindgen** по `core/include/triton/core/tritonserver.h`
  (значит нужен `libclang` — в devShell он есть, `LIBCLANG_PATH` задан в `flake.nix`) и выдаёт
  только `cargo:rustc-link-lib=dylib=tritonserver`, **без** `rustc-link-search`: путь к ядру
  надо передать самому — `RUSTFLAGS="-L $PWD/third_party/triton-runtime/lib"` при сборке с
  фичей `triton-live` (в рантайме — `LD_LIBRARY_PATH` на тот же каталог, `tools/triton_env.sh`).
- Побочный эффект `build.rs`: при отсутствии заголовка он сам делает
  `git submodule update --init --recursive` — в песочнице это лишний шаг, но вреда не несёт
  (сабмодуль `core` уже выкачан, план Шаг 1).

#### Task 2.3 — сделано (движок работает)

`crates/madmom-ml/src/triton.rs`: `TritonConfig` + `TritonModel` (`ActivationModel`).
Сервер стартует, ONNX-бэкенд грузит модель, запрос проходит, активации возвращаются.
Тест `crates/madmom-ml/tests/triton_live.rs` (фича `triton-live`) — зелёный.

Что выяснилось и учтено:

- биндинги захватывают `tokio::runtime::Handle::current()` при старте сервера, поэтому
  модель владеет `Runtime` и все вызовы идут через него (`futures::executor::block_on`
  недостаточно), и рантайм обязан жить дольше сервера;
- `Control::Explicit` сам ничего не грузит: модель запрашивается явно (`load_model`) и
  затем опрашивается до готовности, иначе первый запрос падает `unknown model`;
- бэкенд ONNX Runtime жёстко ставит `INSTALL_RPATH "$ORIGIN"` (его CMakeLists, строки
  269/280), так что `-DCMAKE_INSTALL_RPATH` не действует и линковки мало: сборочный
  скрипт кладёт `libonnxruntime.so*` рядом с бэкендом, куда `$ORIGIN` и указывает;
- `tritonserver-rs` исключён из workspace: path-зависимость внутри каталога workspace —
  неявный участник, из-за чего любая сборка `--workspace` тянула его с его default-фичами
  (`gpu` → bindgen, cuda-driver-sys) и clippy падала на его build-скрипте;
- сборка с фичей требует `RUSTFLAGS="-L $PWD/third_party/triton-runtime/lib64"` и
  `LD_LIBRARY_PATH` на тот же каталог: bindings выдают только `cargo:rustc-link-lib`
  без search path.

**Задержка (шаг 6 Task 2.3):** batch=1 — 1.69 мс/вызов (591 кадр/с), batch=10 — 0.23 мс/кадр, batch=100 — 0.054 мс/кадр. Для online-режима при 100 fps бюджет 10 мс/кадр, то есть одиночный вызов укладывается с ~6-кратным запасом и батчинг не обязателен.

**Проверка на настоящей модели** (не на примере): `chroma_dnn`, 20 кадров по 1575
признаков, входные данные одни и те же — движок против madmom:
`max_abs_diff = 1.6e-07`, `cosine = 1.0000000`. По дороге нашлись три дефекта
конвертера (IR-версия 13 против максимума 10 у вендорного ORT; ширина входа из
таблицы вместо весов; выход всегда 1 вместо 12) — исправлены, коммит `8662cda`.

#### ONNX-графы рекуррентных семейств — сделано

`tools/onnx_recurrent.py`: каждый слой — один `Scan`, тело считает один шаг из предыдущего
состояния, ровно в порядке `madmom/ml/nn/layers.py` (i/f читают прошлую ячейку, o — новую;
GRU: `h = ug*cell + (1-ug)*h_prev`). Двунаправленные слои идут тем же телом с обратным
направлением `Scan`, выход разворачивается обратно (явный `Range`/`Gather`-реверс ORT не
переваривает: «Invalid dim0_offset of 0»).

Ловушки, видные только на реальных моделях:

- старые GRU-пиклы хранят начальное состояние как `hid_init`; madmom переименовывает его в
  `__setstate__`, а заглушки конвертера — нет. Чтение только `init` давало старт с нулей:
  расхождение 6e-2 при внешне правдоподобном выходе;
- `config.pbtxt` расходился с графом по ширине выхода (`notes_brnn` объявлял 1, у графа 88):
  для рекуррентных семейств ширина выхода тоже берётся из последнего слоя.

Сверка через движок против madmom на фиксированных 50 кадрах, cosine 1.0000000 у всех пяти:
`onsets_rnn_1` 2.7e-7, `notes_brnn` 1.6e-6, `downbeats_bgru_harmonic_0` 4.8e-7,
`beats_lstm_1` 8.2e-8, `beats_blstm_1` 6.7e-8.

Полная конвертация: **80 из 92** (было 1). Осталось 12: `tcn` 8, `crf` 2, `gmm` 2.

#### Свёрточные (`cnn`) — построитель есть, сверена одна модель

`tools/onnx_conv.py`: данные один раз переводятся в NCHW, свёртки с паддингом/шагом madmom,
при `Reshape`/`FeedForward` — переход в `(кадры, бины * каналы)`. Поддержаны Conv, BatchNorm
(как scale/shift), MaxPool (`maximum_filter` + срез madmom = valid-pool), Pad, Transpose,
Average, Reshape, Stride.

**Главная ловушка:** `scipy.ndimage.convolve` — свёртка (переворачивает ядро), ONNX `Conv` —
корреляция. Без переворота граф работает и молча отдаёт зеркальные признаки; закреплено тестом
`tools/tests/test_conv.py`.

Численно сверено: `chords_cnnfeat` 2.0e-06 (cos 1.000000), `key_cnn` 9.5e-07 (cos 1.0, на
выходе `(1, 24)` — сеть усредняет кадры и отвечает один раз на вход). **Гейт типов (обязателен).** Конвертер теперь прогоняет `onnx.shape_inference` в strict-режиме и
считает модель `unsupported`, если граф не проходит проверку типов; ошибки внутри `Scan`
игнорируются (динамическая ось кадров у `downbeats_blstm_*` даёт ложное расхождение, но граф
рабочий). Без этого гейта список «converted» врал: `key_cnn_1..4` и `onsets_cnn` попадали в него
с несовместимыми формами. Теперь **72 converted / 20 unsupported**, и все конвертированные либо
сверены числами, либо однородны со сверенными.

**НЕ сверены и неверны (гейт их отсекает):** `key_cnn_1..4` (2017). Они не «та же архитектура»:
их цепочка — Pad/Conv×5, `TransposeLayer`, `ReshapeLayer`, `FeedForward`, `AverageLayer`,
`FeedForward`, то есть ранг меняется в середине, и мой построитель даёт граф, который падает на
`ReduceMean` по оси 2 при 2-D тензоре. Статус `converted` для них — ложный: числа не совпадают
(проверить не удалось, граф не исполняется). То же, что было с `notes_cnn`: построитель не
должен рапортовать успех для структуры, которую не понимает и `onsets_cnn` — он устроен иначе, и построитель
его пока не покрывает. Разбор его цепочки (8 слоёв):

```
0 BatchNormLayer   mean/inv_std (80, 3), gamma=1, beta=0  -> нормировка по (бины, каналы),
                                                            а не по каналам: scale/shift
                                                            имеют форму (1, C, 1, B)
1 ConvolutionalLayer (3, 10, 7, 3), bias (10,)
2 MaxPoolLayer     size (2,), stride (2,)                 -> ОДНОМЕРНЫЙ size; madmom берёт
                                                            size[1] и на 3-Д данных это
                                                            отдельная ветка, которую надо
                                                            воспроизвести точно
3 ConvolutionalLayer (10, 20, 3, 3)
4 MaxPoolLayer     size (2,), stride (2,)
5 StrideLayer
6 FeedForwardLayer (1120, 256)
7 FeedForwardLayer (256, 1)
```

То есть для него нужны: BN с формой `(бины, каналы)`, MaxPool с одномерным `size` и корректная
связка `Stride -> FeedForward(1120)`, где 1120 = кадры_после_стрида × 80 × 10. Нынешний
построитель даёт на нём неверные формы (`Mul` с несовместимыми размерностями) — это ожидаемо,
а не случайность.

**Отдельная находка:** `notes_cnn`, `notes_cnn_1`, `notes_cnn_2` — не сети, а
`SequentialProcessor` (свёртки + `ParallelProcessor`). Построитель теперь отказывает для всего,
что не `NeuralNetwork`: раньше он читал `.layers` у обёртки и молча строил другой граф, помечая
модель как converted. Это была бы тихая ошибка в числах.

Входные формы (замерены по процессорам madmom, а не по таблице): `chords_cnnfeat` (113, 1),
`key_cnn*` (105, 1), `notes_cnn*` (144, 1), `onsets_cnn` (80, **3**). Таблица `FEATURE_WIDTHS`
была неверна для всех них (стояло 3), поэтому введена отдельная `CNN_INPUTS`.

#### Task 2.3: Triton in-process (`tritonserver-rs` 0.4.1)

**Files:** Create `crates/madmom-ml/src/triton.rs`; Test `crates/madmom-ml/tests/triton_contract.rs`.

**Interfaces (Produces):**
```rust
pub struct TritonModel {                     // реализует ActivationModel (D1)
    pub server: Arc<Server>,                 // один Server на процесс (D10)
    pub model_name: String, pub model_version: i64,   // -1 = последняя
    pub input_name: String, pub output_name: String,
    pub shape: InputShape,
    pub runtime: tokio::runtime::Runtime,    // синхронный фасад: block_on(infer_async)
}
pub fn start_server(cfg: &TritonConfig) -> Result<Arc<Server>>;   // Options + Server::new().await
// madmom.toml: [triton] repository / backend_directory / model_version / control / cpu_only
```

- [ ] Шаг 1: зависимость (сабмодуль в нашем репозитории):
      `tritonserver-rs = { path = "../../third_party/tritonserver-rs", default-features = false }`
      (из `crates/madmom-ml/Cargo.toml`; CPU-only — решение владельца, фича `gpu` не включается).
      `third_party/tritonserver-rs` — сабмодуль (`github.com/UncleAndy/tritonserver-rs`, 0.4.1),
      его вложенный `core` (r24.08, C-API-заголовки) уже выкачан; для чистого клона —
      `git submodule update --init --recursive`. В devShell — `libclang` (`LIBCLANG_PATH`),
      либа Triton и бэкенды — см. R16.
- [ ] Шаг 2: `start_server`: `Options::new(repo)` (⚠ `.backend_directory(...)` **в биндингах
      отсутствует** — см. «Сборка рантайма» выше: либо симлинк `/opt/tritonserver/backends`,
      либо правка биндингов по решению владельца),
      `.model_control_mode(Control::Explicit)`, `.startup_model(name)`,
      `.strict_model_config(true)`, `.exit_timeout(Duration::from_secs(5))`,
      `.pinned_memory_pool_byte_size(0)`; `Server::new(opts).await`.
- [ ] Шаг 3: один инференс (шаблон — `tritonserver-rs/examples/src/bin/audio_classification.rs`):
      `server.create_request(name, -1)` → `request.add_default_allocator()` →
      `request.add_input_with_dims(input_name, Buffer::from(slice), dims)?` →
      `runtime.block_on(request.infer_async()?)` → `Response::get_output(output_name)`
      → `Buffer::as_ref::<f32>()` → `ArrayD` нужной формы; ошибки →
      `MadmomError::Inference { model, cause }`.
      Ожидаемая форма `config.pbtxt` для наших моделей (по образцу `yamnet`):
      `backend: "onnxruntime"`, `input { name: "..." data_type: TYPE_FP32 dims: [-1, <features>] }`
      (`-1` — переменное число кадров; для CNN/TCN `dims: [-1, <freq>, <channels>]`),
      `output { name: "..." data_type: TYPE_FP32 dims: [-1, <out>] }`, `max_batch_size: 0`
      (батчинг по кадрам делает Rust, не Triton).
- [ ] Шаг 4 (тест): `server.is_ready()` / `model_is_ready(name, -1)`; отсутствие
      репозитория → понятная ошибка, а не паника.
- [ ] Шаг 5: ансамбли: 8 моделей — 8 запросов через `futures::future::join_all`
      в одном `block_on` (порядок усреднения фиксирован, D9).
- [ ] Шаг 6: online-режим (1 кадр за вызов, 100 fps) — замер задержки; при провале
      батчить N кадров (DBN-путь работает по forward-переменным и терпим к задержке).
- [ ] Шаг 7: коммит `feat(ml): triton in-process inference (tritonserver-rs)`.

#### Task 2.4: Приёмка ONNX-репозитория

**Files:** Create `crates/madmom-ml/tests/onnx_repo.rs`; Test — требует `MADMOM_MODEL_REPO`.

- [ ] Шаг 1 (тест): по спецификации §2.1 пройти по всем моделям репозитория и сверить:
      имя/версия, имя и dtype входного и выходного тензора, форму входа; отсутствие
      лишних/недостающих моделей (ожидаемый список — в §2.1).
- [ ] Шаг 2 (тест): для каждой сети ансамбля подать эталонные признаки
      (`tests/golden/preproc/*.npy`) и сверить выход с эталонным выходом **этой** сети
      (`tests/golden/activations/<model>.npy`, снимается `tools/dump_preproc.py`),
      допуск §4.2 (float32, `1e-4`). Это ловит ошибки конвертации по-модельно, до
      сборки ансамбля.
- [ ] Шаг 3 (тест): CPU-only — сервер стартует без CUDA (`pinned_memory_pool_byte_size(0)`,
      `KIND_CPU`), `server.model_is_ready(name, -1)` истинно для всех моделей.
- [ ] Шаг 4: коммит `test(ml): onnx repository acceptance`.

#### Task 2.5: Паритет активаций через Triton (ONNX)

- [ ] Шаг 1 (тест, feature `triton-live`): `RNNBeatProcessor` = `sample.beats_blstm.npz`
      (`max|Δ| ≤1e-4`), `RNNOnsetProcessor` = `sample.onsets_brnn.npz`,
      `TCNBeatProcessor` = `sample.beats_tcn_beats.npz`, `DeepChromaProcessor` = `sample.deep_chroma.npz`.
- [ ] Шаг 2: расхождение здесь — дефект конвертации модели или контракта входа, а не
      «допуск»: сначала сверять `tests/golden/preproc/*.npy`, потом сеть.
- [ ] Шаг 3: коммит `test(features): live triton activation parity`.

---

### Фаза 3 — HMM-ядро и «флагманские» пайплайны (биения, онсеты, темп)

**Статус (2026-11-15): биения в golden-режиме работают.** Сделано:

- **Task 3.1 (частично)** — `crates/madmom-ml/src/hmm.rs`: `TransitionModel`
  (CSR-подобное разреженное представление, `make_sparse`/`from_dense`, логарифмы
  вероятностей считаются один раз при построении), трейт `ObservationModel`,
  `DiscreteObservationModel` (таблица `(состояния × наблюдения)`, как в эталоне),
  `HiddenMarkovModel::viterbi` в лог-домене. Проверено доктестом самого madmom
  (`-12.87…`, путь `[1,1,0,0,1,1,1,0,0]`) и тестами на разреженную раскладку, отказ при
  не-распределении и `-inf` → пустой путь. **Не сделано:** `forward`/`forward_step`/`reset`
  (нужны онлайн-режиму), `make_dense`.
- **Task 3.2 (частично)** — `crates/madmom-features/src/beats_hmm.rs`: `BeatStateSpace`
  (линейные и лог-интервалы с итеративной дедупликацией), `exponential_transition`
  (порог `np.spacing(1)`, нормировка строк, `lambda=None` → диагональ),
  `BeatTransitionModel`, `RNNBeatTrackingObservationModel`. **Не сделано:** `BarStateSpace`,
  `MultiPatternStateSpace`, `BarTransitionModel`, `MultiPatternTransitionModel`,
  `RNNDownBeatTrackingObservationModel`, `GMMPatternTrackingObservationModel`.
- **Task 3.4 (частично)** — `crates/madmom-features/src/beats.rs`: `threshold_activations`,
  `DBNBeatTrackingProcessor` (константы эталона 55/215/100/16/0/true) с `process_offline`
  (Viterbi → регионы beat-range → пик активации → секунды). **Не сделано:** онлайн-режим
  (`forward`+`argmax`), `correct=false` (`argrelmin` из scipy), `BeatTrackingProcessor`,
  `BeatDetectionProcessor`, `CRFBeatDetectionProcessor`.

**Статус downbeats (2026-11-15): тоже в golden-режиме, тоже без Triton.** Сделано в
`crates/madmom-features/src/beats_hmm.rs` (`BarStateSpace` — стопка из `num_beats`
битовых пространств с накопительными позициями, `BarTransitionModel` — смена темпа
только на границе доли, `RNNDownBeatTrackingObservationModel` — три колонки
наблюдений) и `crates/madmom-features/src/downbeats.rs`
(`DBNDownBeatTrackingProcessor` с константами эталона `NUM_TEMPI=60`, `THRESHOLD=0.05`,
`OBSERVATION_LAMBDA=16`, `TRANSITION_LAMBDA=100`; строка побеждает по лог-вероятности
пути). **Не сделано:** `MultiPatternStateSpace`/`MultiPatternTransitionModel`,
`RNNDownBeatTrackingObservationModel` на нескольких паттернах и CRF-варианты.

Главная сверка (golden-режим: активации из эталонного дампа, сеть не нужна):

```
путь Витерби: 281 из 281 кадров идентичны эталону (точные целые состояния)
log probability: -758.193325 против -758.193326 в дампе
биения: [0.1, 0.45, 0.8, 1.12, 1.48, 1.8, 2.15, 2.49] — все в пределах 1e-6
```

Сверка downbeats на активациях эталона (`sample.downbeats_blstm.npz`, 281 кадр × 2
колонки):

```
путь Витерби 3/4: 281 из 281 кадров идентичны, log prob -764.586594
путь Витерби 4/4: 281 из 281 кадров идентичны, log prob -760.212077   ← победившая строка
пути после порога 0.05 (строки 7..267): 261 из 261 идентичны
(down-)beats: [(0.09,1) (0.45,2) (0.79,3) (1.12,4) (1.47,1) (1.8,2) (2.14,3) (2.49,4)]
```

Две ловушки, которые дамп вскрыл сразу: `np.argmax` по окну из двух колонок —
**плоский** (строка с меньшей первой колонкой выигрывает второй: 0.08 с вместо 0.09 с),
а madmom возвращает `np.vstack(...).T`, поэтому `np.save` пишет массив в
**Fortran-порядке** — читатель `.npy` теперь разворачивает его при чтении.

Это первый сквозной результат: `WAV → кадры → STFT → фильтрбанк → логи-спектр →
разности → (активации из фикстуры) → DBN → биения / сильные доли`. Осталось подключить
саму сеть (Фаза 2, Triton на хосте) — и флагманский пайплайн биений будет полным.


#### Task 3.1: `ml::hmm`

**Files:** Create `crates/madmom-ml/src/hmm.rs`; Test `…/tests/hmm.rs`
(порт `tests/test_ml_hmm.py`, 9 тестов + doctest).

**Interfaces (Produces):** `TransitionModel { states, pointers, probabilities }`,
`make_sparse/make_dense/from_dense`, `ObservationModel` (trait),
`DiscreteObservationModel`, `HiddenMarkovModel::{viterbi, forward, forward_step, reset}`.

- [x] Шаг 1 (тест): doctest `TransitionModel.from_dense([0,1,0,1],[0,0,1,1],[0.7,0.3,0.6,0.4])`
      + `DiscreteObservationModel([[0.2,0.3,0.5],[0.7,0.1,0.2]])` → `viterbi([0,0,1,1,0,0,0,2,2])`
      = `[1,1,0,0,1,1,1,0,0]`, `log_p ≈ -12.87`; таблица `forward` из докстринга.
- [x] Шаг 2 (тест): `make_sparse` поднимает `ValueError`, если распределение не
      суммируется в 1; CSR-инвариант: `pointers[state]..pointers[state+1]` покрывает
      все переходы *в* `state`.
- [x] Шаг 3 (тест): Viterbi с `-inf` → предупреждение и пустой путь (`is_empty`).
- [ ] Шаг 4: коммит `feat(ml): HMM viterbi/forward core`.

#### Task 3.2: Пространства состояний и модели наблюдений биений

**Files:** Create `crates/madmom-features/src/beats_hmm.rs`;
Test `…/tests/beats_hmm.rs` (порт `tests/test_features_beats_hmm.py`, 17 тестов).

**Interfaces (Produces):** `BeatStateSpace`, `BarStateSpace`,
`MultiPatternStateSpace`, `BeatTransitionModel`, `BarTransitionModel`,
`MultiPatternTransitionModel`, `RNNBeatTrackingObservationModel`,
`RNNDownBeatTrackingObservationModel`, `GMMPatternTrackingObservationModel`,
`exponential_transition`.

- [x] Шаг 1 (тест): `BeatStateSpace(40, 400)` — `num_states`, `num_intervals`,
      `first_states`, `last_states`, лог-шкала при `num_intervals < len(linear)`.
- [x] Шаг 2 (тест): `exponential_transition` — `np.spacing(1)`-порог,
      нормировка по строкам, `transition_lambda=None` → диагональная матрица.
- [x] Шаг 3а (тест): `BarStateSpace(4, …)` — накопительные позиции, offsets
      `first_states`/`last_states` совпадают с эталоном (Unit-тест на этой стороне).
- [ ] Шаг 3б: `MultiPatternStateSpace` и `MultiPatternTransitionModel` — не сделано.
- [x] Шаг 4: коммит `feat(features): beat/downbeat HMM state and observation models`.

#### Task 3.3: Пик-пикинг онсетов

**Files:** Create `crates/madmom-features/src/onsets.rs` (частично: `peak_picking`,
`OnsetPeakPickingProcessor`, 11 ODF-функций);
Test `…/tests/onsets.rs` (порт `tests/test_features_onsets.py`, 31 тест).

**Статус (2026-11-15): ODF и пик-пикинг сделаны и сверены.** Файлы:
`crates/madmom-features/src/onsets.rs`, `crates/madmom-core/src/{ndimage,utils,scipy}.rs`
(фильтры scipy.ndimage, `combine_events`, `argrel*`), `crates/madmom-audio/src/stft.rs`
(`local_group_delay` + `np.unwrap`), `crates/madmom-audio/src/spectrogram.rs`
(`diff_max_bins` — раньше отказывал, теперь реализован через maximum_filter).

- [x] Шаг 1 (тест): 11 ODF-функций — сделано, сверено с 11 дампами (макс. 4.3e-06, у
      комплексно-доменных 1.2e-04: эталон считает `exp(1j*phase)` в complex64).
- [x] Шаг 2 (тест): `peak_picking` — сделано; кадры пиков совпадают **точно** на всех
      четырёх кейсах (в т.ч. скользящие mean/max с `origin` и `mode='constant'`).
- [x] Шаг 3 (тест): `OnsetPeakPickingProcessor` offline и online — сделано; все onset'ы
      совпадают точно, `process_online` (всё сразу и покадрово) = offline.
- [x] Шаг 4: коммит `feat(features): onset detection functions and peak picking`.

**Сверка (21 дамп, `tools/dump_onsets.py` → `tests/golden/onsets/`):**

```
вход:  spec 7.629e-06 (1 ulp f32), log_filt_spec 1.341e-07, phase 1.192e-07 (завёрнутая)
ODF:   hfc 4.6e-07, spectral_diff 5.2e-07, spectral_flux 2.0e-06, superflux 1.6e-06,
       complex_flux 4.3e-06, mkl 1.3e-07, phase_deviation 1.0e-07, wpd 1.6e-07,
       normalized_wpd 1.7e-07, complex_domain 1.2e-04, rectified 1.0e-04
пики:  35/40/16/24 кадров — совпадение точное
onset'ы: offline 16, online 16 (всё сразу) и 16 (покадрово) — разница 0.000e0
```

Ловушки, вскрытые дампами: `phase_deviation` и `complex_flux` **не** должны требовать
совпадения формы фазы со спектрограммой (эталон берёт фазу из неотфильтрованного STFT),
а фазово-взвешенные ODF, наоборот, обязаны отказывать на отфильтрованной; у
`_complex_domain` цель первого кадра — голая магнитуда (без своей фазы).

**Не сделано:** ODF-процессоры, оборачивающие сеть (`SpectralOnsetProcessor`,
`RNNOnsetProcessor`, `CNNOnsetProcessor`, `MultiOnsetProcessor`) — они относятся к фазе
инференса; `correlation_diff` (эталон сам называет её «only for completeness», в тестах
не используется).

#### Task 3.4: DBN- и CRF-трекеры биений

**Files:** Create `crates/madmom-features/src/beats.rs`, `…/beats_crf.rs`;
Test `…/tests/beats.rs` (порт `tests/test_features_beats.py`, 11 тестов).

**Константы (обязательны к точному переносу):** `DBNBeatTrackingProcessor`:
`MIN_BPM=55`, `MAX_BPM=215`, `NUM_TEMPI=None`, `TRANSITION_LAMBDA=100`,
`OBSERVATION_LAMBDA=16`, `THRESHOLD=0`, `CORRECT=True`;
`BeatTrackingProcessor`: `LOOK_ASIDE=0.2`, `LOOK_AHEAD=10`;
`CRFBeatDetectionProcessor`: `INTERVAL_SIGMA=0.18`, `FACTORS=[0.5,0.67,1.0,1.5,2.0]`,
`NUM_INTERVALS=5`, `MIN_BPM=20`, `MAX_BPM=240`, `ACT_SMOOTH=0.09`, `HIST_SMOOTH=7`.

**Статус (2026-11-15): сделано полностью и сверено.** Файлы:
`crates/madmom-features/src/beats.rs` (`detect_beats`, `BeatTrackingProcessor`,
`BeatDetectionProcessor`, `correct=false`, онлайн-режим), `…/beats_crf.rs` (CRF целиком),
`crates/madmom-ml/src/hmm.rs` (`forward`/`reset`/линейные `densities`),
`crates/madmom-core/src/ndimage.rs` (`correlate1d`), `crates/madmom-audio/src/signal.rs`
(`signal_frame_f64`).

- [x] Шаг 1 (тест): `threshold_activations` (первый/последний кадр ≥ threshold).
- [x] Шаг 2 (тест): `DBNBeatTrackingProcessor` offline (`correct=True`: пик внутри
      beat-range) и `correct=False` (`argrelmin(state_positions[path], mode='wrap')` +
      проверка `pointers[path[beats]] == 1`); онлайн: `forward` + `argmax` + правило
      `next_beat = last_beat + 60/max_bpm` (HMM `forward` нормализует по кадрам в линейном
      домене, как эталон).
- [x] Шаг 3 (тест): сверка с `sample.dbn_beat_tracker.txt` — **побитово**: вывод DBN на
      активациях фикстуры совпадает с файлом командной программы (8/8 биений).
- [x] Шаг 4 (тест): `BeatTrackingProcessor` (локальный темп) = `tracker`;
      `BeatDetectionProcessor` (глобальный темп) = `detector`;
      `CRFBeatDetectionProcessor` = `crf` (+ его кандидатные интервалы).
- [x] Шаг 5: коммит `feat(features): beat trackers (DBN, tracking, CRF)`.

**Сверка (17 дампов, `tools/dump_beats.py` → `tests/golden/beats/`):**

```
dbn_default 8, det_dbn 8 (тот же файл), dbn_edge_frames 5, dbn_no_correction 8,
dbn_threshold_025 7, dbn_threshold_1 0, dbn_empty_path 0,
dbn_forward_all 5, dbn_forward_frames 5, dbn_forward_frames_noreset 5,
tracker 8, detector 8, crf 3, crf_intervals 5 — все в пределах 1e-6
```

Ловушка в читателе `.npy`, вскрытая дампом: пустой результат трекера пишется как массив
`(0,)`, а счётчик элементов брал `.max(1)` → чтение требовало 8 байт, которых нет
(`BufferTooShort { need: 136, have: 128 }`). Исправлено: скаляр (0-d) — один элемент,
массив с нулевой размерностью — ни одного.

#### Task 3.5: Оценка темпа

**Files:** Create `crates/madmom-features/src/tempo.rs`;
Test `…/tests/tempo.rs` (порт `tests/test_features_tempo.py`, 37 тестов).

**Константы:** `METHOD='comb'`, `ALPHA=0.79`, `MIN_BPM=40`, `MAX_BPM=250`,
`ACT_SMOOTH=0.14`, `HIST_SMOOTH=9`, `HIST_BUFFER=10.0`, `NO_TEMPO=NaN`.

**Статус (2026-11-15): офлайн-часть сделана, сверена с эталоном.**

- [x] Шаг 1 (тест): `interval_histogram_acf`, `interval_histogram_comb` (банк гребёнчатых
      фильтров, `feed_backward`), `dominant_interval` — сделано
      (`crates/madmom-audio/src/comb_filters.rs` + `crates/madmom-features/src/tempo.rs`).
- [x] Шаг 2 (тест): `detect_tempo` — сделано **без** `interpolate=True`; `argrelmax`
      (`mode='wrap'`) перенесён в `madmom-core::scipy` и сверен с scipy напрямую (строгое
      `>`, поэтому плато не даёт пика; заворот концов). `interpolate=True` **отложено и
      отказывает типизированной ошибкой** (`TempoError::InterpolationNotPorted`): нужна
      квадратичная B-сплайн интерполяция scipy (`interp1d(..., 'quadratic')`,
      not-a-knot). Дампы для неё сняты, разрыв измерим.
- [x] Шаг 3 (тест): `TempoEstimationProcessor` (офлайн), `CombFilter/ACF/DBN/TCN`
      histogram processors, `TCNTempoHistogramProcessor` — сделано. Онлайн-режим
      (`process_online`, буферы, `hist_buffer`) **отложен**.
- [x] Шаг 4: коммит `feat(features): tempo estimation`.

**Сверка (26 дампов, `tools/dump_tempo.py` → `tests/golden/tempo/`):**

```
comb-гистограмма        8.882e-16   (после округления alpha до f32 — иначе 4.378e-07)
acf-гистограмма         3.331e-16
DBN-гистограмма         0.000e+00   (счёт интервалов по пути Витерби, всего 281)
tempi (comb/acf/dbn)    ≤1.110e-16
TCN: гистограмма и tempi — совпадение, 14 и 9 строк
```

Ловушка, которую вскрыл дамп: в Cython-хелпере эталона `alpha` объявлен как C `float`,
то есть гребёнчатый фильтр работает с **округлённым до f32** множителем (сигнал при этом
остаётся float64), а `feed_forward` — чистый numpy и остаётся f64.

---

### Фаза 4 — остальные MIR-задачи

Порядок внутри фазы — по зависимости моделей: downbeats → notes → key → chords → TCN/MM → GMM.

| # | задача | файлы (Rust) | тесты-источники | результат (фикстура) |
|---|---|---|---|---|
| 4.1 | downbeats: `RNNDownBeatProcessor`, `DBNDownBeatTrackingProcessor`, `RNNBarProcessor`, `DBNBarTrackingProcessor`, `LoadBeatsProcessor`, `SyncronizeFeaturesProcessor` | `madmom-features/src/downbeats.rs` | `tests/test_features_downbeats.py` (15) | `sample.dbn_downbeat_tracker.txt`, `sample.bar_tracker.npz` |
| 4.2 | notes: `RNNPianoNoteProcessor`, `CNNPianoNoteProcessor`, `NoteOnsetPeakPickingProcessor`, `ADSRNoteTrackingProcessor` + `notes_hmm` (ADSR-пространство состояний/переходы/наблюдения) | `madmom-features/src/{notes,notes_hmm}.rs` | `tests/test_features_notes.py` (5) | `stereo_sample.piano_transcriptor.txt` |
| 4.3 | chroma/deep chroma: `DeepChromaProcessor`, `CLPChromaProcessor` | `madmom-audio/src/chroma.rs` (дописать) | `tests/test_audio_chroma.py` (3) | `sample.deep_chroma.npz` |
| 4.4 | key: `CNNKeyRecognitionProcessor`, `key_prediction_to_label`, 24 метки `KEY_LABELS` | `madmom-features/src/key.rs` | `tests/test_features_key.py` (2) | `sample.key_recognition.txt`, `sample2.key_recognition.txt` |
| 4.5 | chords: `CNNChordFeatureProcessor` (супер-кадры/усреднение `_cnncfp_superframes/_avg`), `CRFChordRecognitionProcessor`, `DeepChromaChordRecognitionProcessor`; CRF — **из ONNX** (решение 3) | `madmom-features/src/chords.rs` | `tests/test_features_chords.py` (6), `tests/test_ml_crf.py` (1) | `sample.cnn_chord_recognition.txt`, `sample.dc_chord_recognition.txt`, `sample2.*` |
| 4.6 | GMM-трекер паттернов: `PatternTrackingProcessor` + `ObservationSource` (GMM — **из ONNX**, log-плотности batch-вызовом) | `madmom-features/src/downbeats.rs` | `tests/test_features_downbeats.py` | `sample.gmm_pattern_tracker.txt`, `sample.gmm_pattern_tracker.npz` |
| 4.7 | multi-model: `MultiModelSelectionProcessor`, `MMBeatTracker` | `madmom-features/src/beats.rs` | `tests/test_bin.py` | `sample.mm_beat_tracker.txt` |

Задачи 4.5 (CRF-аккорды) и 4.6 (GMM-паттерны) требуют ONNX-обёрток (решение 3): граф
вокруг матриц CRF и вокруг log-плотностей GMM. Математика `ml/crf.py` и `ml/gmm.py` в
Rust не портируется — наш объём только контракт входа/выхода и приёмка (см. R13, R15).

Каждая задача: тесты из указанного файла madmom переносятся 1:1 (+ doctest-числа),
сверка с фикстурой, коммит `feat(features): <задача>`.
Отдельно проверить: `DBNDownBeatTrackingProcessor` (`NUM_TEMPI=60`, `THRESHOLD=0.05`,
`MIN/MAX_BPM=55/215`), `PatternTrackingProcessor` (`MIN_BPM=(55,60)`,
`MAX_BPM=(205,225)`), `DBNBarTrackingProcessor` (`OBSERVATION_WEIGHT=100`,
`METER_CHANGE_PROB=1e-7`) — числа берутся из исходников, не из головы.

---

### Фаза 5 — io, CLI, MIDI

#### Task 5.1: Текстовые события (`io::events`)

**Files:** Create `crates/madmom-io/src/events.rs`;
Test `…/tests/events.rs` (порт `tests/test_io.py`, 22 теста).

- [ ] Тесты: `load_events/write_events/load_beats/write_beats/load_downbeats/
      load_notes/write_notes/load_segments/write_segments/load_key/write_key/
      load_tempo/write_tempo` — форматы: комментарии `#`, секунды с 3 знаками,
      ноты `onset\tpitch\tduration`, сегменты `SEGMENT_DTYPE`, тональности;
      `Encoding='utf8'`, работа с file handle и путём.

#### Task 5.2: `.npz`-активации (`features::activations`)

**Files:** Create `crates/madmom-features/src/activations.rs`;
Test `…/tests/activations_io.rs`.

- [ ] Тесты: `Activations::load` (npz с `fps`, npz без `fps`, текстовый файл с
      `sep`, массив), `ActivationsProcessor` (`mode='r'|'w'`, `fps`, `sep`),
      `data.ndim>1 && shape[1]==1` → flatten.

#### Task 5.3: MIDI

**Files:** Create `crates/madmom-io/src/midi.rs`, `crates/madmom-core/src/midi.rs`
(порт `madmom/utils/midi.py` — 35 классов событий, `MIDIFile`, `MIDITrack` — и
`madmom/io/midi.py`);
Test `…/tests/midi.rs` (порт `tests/test_utils_midi.py` + `tests/test_io_midi.py`, 18 тестов).

- [ ] Шаг 1 (тест): парсинг/запись `tests/data/annotations/*.mid`
      (в т.ч. sustain, `multitrack.mid`), `tick2second/second2tick/tick2beat/beat2tick`.
- [ ] Шаг 2 (тест): доктест-числа `utils/midi.py` (26 примеров).
- [ ] Шаг 3: бэкенд — **`midly`** (решение владельца) + переходный слой событий,
      совместимый с `madmom.utils.midi` (`NoteOn/NoteOff/SetTempo/TimeSignature/
      KeySignature/SmpteOffset/…`). Паритет проверяется на корпусе
      `tests/data/annotations/*.mid`; байт-в-байт round-trip не требуется, но набор
      событий и тайминги (`tick2second`, sustain) обязаны совпасть.

#### Task 5.4: CLI (`madmom-cli`, 24 бинаря в первом релизе)

**Files:** Create `crates/madmom-cli/src/bin/*.rs`, `…/src/common.rs`;
Test `crates/madmom-cli/tests/*.rs` (порт `tests/test_bin.py`, 105 тестов).

- [ ] Шаг 1: общий слой: подкоманды `single|batch|online`, `-j/--num_threads`,
      `-o`, `-s/--suffix`, `--shuffle`, `--load/--save` активаций, `-v`.
- [ ] Шаг 2: бинари: `BarTracker BeatDetector BeatTracker CNNChordRecognition
      CNNOnsetDetector ComplexFlux CRFBeatDetector DBNBeatTracker DBNDownBeatTracker
      DCChordRecognition GMMPatternTracker KeyRecognition LogFiltSpecFlux MMBeatTracker
      OnsetDetector OnsetDetectorLL PianoTranscriptor SpectralOnsetDetection SuperFlux
      SuperFluxNN TCNBeatTracker TCNTempoDetector TempoDetector` (24 бинаря).
      `evaluate` отложен вместе с `madmom-eval` (решение 4), `PickleProcessor` не
      портируется (D3) — из 25 программ madmom в первом релизе 24.
- [ ] Шаг 3: end-to-end тест: каждый бинарь на `sample.wav` = соответствующий
      `tests/data/detections/*.txt` (побайтово, кроме заголовков).
- [ ] Шаг 4: коммит `feat(cli): 24 programs mirroring bin/*`.

---

### Фаза 6 — производительность и публикация (`madmom-eval` отложен)

Метрики (пункты 6.1–6.5) из первого релиза исключены решением 4; в силе остаются
производительность и публикация, а также документирование ограничений.

| # | задача | файлы | тесты-источники |
|---|---|---|---|
| ~~6.1~~ | ~~`evaluation`: `EvaluationMixin`, `SimpleEvaluation`, `Evaluation`, …~~ **отложено (решение 4)** | — | — |
| ~~6.2~~ (отложено) | онсеты/ноты: `onset_evaluation`, `note_onset_evaluation`, `remove_duplicate_notes` | `madmom-eval/src/{onsets,notes}.rs` | `tests/test_evaluation_onsets.py` (15), `…notes.py` (17) |
| ~~6.3~~ (отложено) | биения: `pscore`, `cemgil`, `goto`, `cml`, `continuity`, `information_gain`, `find_longest_continuous_segment` | `madmom-eval/src/beats.rs` | `tests/test_evaluation_beats.py` (56) |
| ~~6.4~~ (отложено) | темп/тональность/аккорды: `tempo_evaluation`, `key_label_to_class`, `score_root/score_exact`, `segmentation`, `reduce_to_triads` | `madmom-eval/src/{tempo,key,chords}.rs` | `…tempo.py` (18), `…key.py` (7), `…chords.py` (23) |
| ~~6.5~~ (отложено) | бинарь `evaluate` + `bin/evaluate` CLI | `madmom-cli/src/bin/evaluate.rs` | `tests/test_bin_evaluate.py` (7) |
| 6.6 | производительность: профиль по стадиям (spectrogram / ONNX-инференс / Viterbi), цель — не медленнее madmom на `sample.wav`, отчёт `benches/` | `crates/*/benches/` | — |
| 6.7 | публикация: `README.md` (лицензии: код BSD, модели CC BY-NC-SA отдельно), описание ONNX-репозитория и требований Triton (R16), doc-тесты, crates.io-метаданные, CI (`.github/workflows/ci.yml`: fmt, clippy, test, golden-parity; `triton-live` — отдельным job в контейнере) | корень | — |

---

## 6. Риски и открытые вопросы

| # | риск | статус / что делать |
|---|---|---|
| R1 | Модели и их конвертация | **вне порта** (D1): Rust моделей не касается. Проверенные факты по pickle сохранены в D1 — для того, кто собирает Triton-репозиторий |
| R2 | Семантика `scipy.ndimage` (свёртка, `maximum_filter`, `uniform_filter`, `mean/min`) | реализовать точечно и тестировать против эталона; таблица в D6 |
| R3 | Порядок суммирования float32 в NN | **снят**: NN-слои в Rust не портируются (D1); на удалённый инференс остаётся допуск §4.2 |
| R4 | `argrelmax(mode='wrap')`, `interp1d('quadratic')`, `np.histogram` с краевыми бинами | возможны расхождения ровно на кадр; тест на фикстурах + явная фиксация причины |
| R5 | Лицензия моделей (CC BY-NC-SA, NonCommercial) | модели не входят ни в крейты, ни в артефакты (D10); размещение — на стороне Triton-репозитория |
| R6 | `Processor.load/dump`, `PickleProcessor`, `argparse`-графы | не портируются (D3); CLI — `clap` |
| R7 | ffmpeg как внешний процесс (не-WAV, ресемплинг, ReplayGain) | сохранить как внешний процесс (D11); тесты с `stereo_sample.m4a`/`_rg.flac` требуют ffmpeg в devShell |
| R8 | `pyaudio`/`Stream` (online-вход с микрофона) | вне первых релизов; `online`-*обработка* (forward) портируется, живой вход — отдельный крейт/задача на `cpal` |
| R9 | `SemitoneBandpassFilterbank` (`ellip`+`filtfilt`) и `SemitoneBandpassSpectrogram` | низкий приоритет, реализуются последними в Фазе 1; тесты не удалять |
| R10 | MIDI: 1400+ строк событий (форк mido) | отдельная задача 5.3; не блокирует музыкальные пайплайны (используется только `PianoTranscriptor --output_format midi` и evaluation) |
| R11 | `sklearn` в `ml/gmm.py::fit` (обучение GMM) | не портируется (инференс — чистый numpy) |
| R12 | Эталон madmom собирается с Cython/opencv/pyfftw; часть тестов может не пройти на этой машине | зафиксировать baseline в Фазе 0 (Task 0.3) и не выдумывать результаты |
| R13 | Готовность ONNX-репозитория моделей (92 сети + CRF + GMM) | спецификация имён/тензоров/ансамблей зафиксирована в §2.1, конвертер — Task 0.8; блокирует только Task 2.4/2.5 и задачи 4.5/4.6; Фазы 3–5 идут в golden-режиме |
| R14 | Задержка online-инференса (1 кадр за вызов при 100 fps) | **замерено, риск снят**: `chroma_dnn` через in-process Triton — 1.69 мс на вызов при batch=1 (591 кадр/с против требуемых 100), 0.23 мс/кадр при batch=10 и 0.054 мс/кадр при batch=100; бюджет кадра при 100 fps — 10 мс, запас ~6x даже без батчинга |
| R15 | Не-NN модели: `ConditionalRandomField` (аккорды) и `GMM` (паттерны) | **решено**: перенос в ONNX (решение 3). Наш объём — контракт (batch на файл; выход: последовательность состояний для CRF, log-плотности для GMM) и приёмка; математика не портируется |
| R16 | Сборка Triton из исходников: `TRITON_CORE_HEADERS_ONLY=OFF` + третьи стороны (gRPC/protobuf/absl/re2/prometheus) тянутся из `triton-inference-server/third_party` при конфигурации; отдельно нужен ONNX Runtime backend и `libclang` для bindgen | **решено: CPU-only + сборка из сабмодуля `core`** (Task 0.7). Остатки: сборка требует сети и времени; герметичная nix-деривация не сделана (сначала devShell-путь + `tools/build_triton_core.sh`); ORT-бэкенд линкуется с системным ONNX Runtime |
| R16a | Герметичность сборки Triton | **решено: вариант D** — сейчас вариант A (разовая сборка в devShell, артефакт в gitignored `third_party/triton-runtime`), а когда/если понадобится CI на чистой машине — C (вендоренные бинарники из NGC-образа с хешами) или B (nix-деривация с предвыборкой входов). Варианты расписаны в §9 |
| R17 | Проверка сборки в этой среде невозможна: `cargo` в sandbox отсутствует, сети нет (crates не скачать), либы Triton нет | сборка и прогоны — на хосте владельца; в плане даны точные команды, выдуманных результатов нет |

## 7. Что не портируется

- `Processor.load/dump`, `pickle_processor`, `bin/PickleProcessor` — сериализация Python-объектов.
- `madmom/__init__.py` (doctest-инфраструктура, `_OutputChecker`, `__version__`).
- `madmom/ml/nn/*` — слои, активации, ансамбль (1454+ строк и Cython-компиляция):
  заменяются ONNX-моделями, инференс — Triton in-process (D1).
- `ml/gmm.py` и `ml/crf.py` как математика — CRF и GMM уезжают в ONNX (D2, R15).
- `ml/gmm.py::GMM.fit`, `ml/nn`-обучение, `hpss`-тренировочные утилиты — только инференс.
- `madmom/evaluation/*` и `bin/evaluate` — отложены, вне первого релиза (решение 4);
  тесты `tests/test_evaluation_*.py` и `tests/test_bin_evaluate.py` тоже отложены.
- `audio/signal.py::Stream`, `process_online` из `processors.py` (живой вход PyAudio).
- Ускорения `pyfftw` и `cv2.filter2D` — заменяются на `rustfft` и одну реализацию свёртки.
- `madmom/utils/stats.py::mcnemar_test` — только если понадобится в `madmom-eval`.
- Тестовые данные не переносятся в репозиторий порта: остаются в `third_party/`
  (submodule) и используются как фикстуры; модели конвертируются в ONNX (Task 0.8)
  в отдельный репозиторий и в крейты не попадают (D1, §2.1).

## 8. Справочные команды

```bash
# 0. Сабмодули (включая вложенные: madmom/models и tritonserver-rs/core)
git submodule update --init --recursive

# 1. Эталон: сборка расширений и baseline тестов (Фаза 0)
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython ps.pytest ps.packaging])' --run \
  'cd third_party/madmom && python3 setup.py build_ext --inplace && python3 -m pytest tests -q'

# 2. Дампы промежуточных стадий эталона (Фаза 0)
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy])' --run \
  'python3 tools/dump_reference.py --audio third_party/madmom/tests/data/audio/sample.wav --out tests/golden'

# 3. Дампы входов/выходов сетей эталона — контракт для ONNX-моделей (Фаза 2)
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy])' --run \
  'python3 tools/dump_preproc.py --out tests/golden/preproc'

# 2b. Triton runtime из исходников (Task 0.7) — один раз, требует сети и времени
./tools/build_triton_core.sh      # libtritonserver.so (CPU-only, теги r24.08)
./tools/build_onnx_backend.sh     # libtriton_onnxruntime.so против системного ORT
source ./tools/triton_env.sh      # LD_LIBRARY_PATH, MADMOM_TRITON_BACKENDS, MADMOM_MODEL_REPO

# 3b. Конвертация моделей в ONNX (по §2.1; источник — 92 .pkl из сабмодуля madmom)
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.onnx])' --run \
  'python3 tools/convert_to_onnx.py \
     --src third_party/madmom/madmom/models \
     --dst third_party/madmom-models-onnx/models'

# 4. Проверка порта (golden-режим, без сети)
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace

# 5. Проверка с Triton in-process (Task 2.5) — нужны libtritonserver + бэкенды (R16)
MADMOM_MODEL_REPO=$PWD/../tritonserver-rs/models \
MADMOM_TRITON_BACKENDS=/opt/tritonserver/backends \
cargo test -p madmom-features --features triton-live

# 6. End-to-end паритет
target/release/BeatTracker single third_party/madmom/tests/data/audio/sample.wav \
  | diff - <(tail -n +2 third_party/madmom/tests/data/detections/sample.beat_detector.txt)
```

## 9. Решения владельца и оставшиеся вопросы

**Принято (2026-11-15):**

1. **Порядок фаз** — «флагманы» (биения / онсеты / темп) первыми (Фаза 3).
2. **MIDI** — `midly` + переходный слой событий (Task 5.3).
3. **Имя публичного крейта** — `madmom-rs`.
4. **Не-NN модели** (CRF, GMM) — в ONNX; контракт (GMM целиком в графе; CRF — граф
   отдаёт `emission/transition/initial/final`, Viterbi в Rust) в §2.2.
5. **NN-инференс** — `tritonserver-rs` **in-process** (bindings к C-API Triton),
   сабмодуль `third_party/tritonserver-rs`; модели — ONNX-репозиторий (§2.1).
6. **CPU-only**: `default-features = false`, `pinned_memory_pool_byte_size(0)`,
   `instance_group KIND_CPU`; GPU — отдельной фичей позже.
7. **`libtritonserver.so` и ONNX Runtime backend — собираем из исходников**
   (сабмодули `third_party/tritonserver-rs/core` и `third_party/onnxruntime_backend`),
   не контейнер. Метрики/статистика/трассировка выключены, теги r24.08 (бэкенд тоже
   переводится на `r24.08` — сейчас он на `main`, что не совпадает с ядром).
   **ONNX Runtime — вендоренный**, системный (nixpkgs) не используется: официальный
   ORT-релиз скачивается один раз с фиксированным URL+sha256 и подключается через
   `TRITON_ONNXRUNTIME_INCLUDE_PATHS`/`_LIB_PATHS`.
8. **Герметичность сборки Triton — вариант D** (решение владельца): сейчас A (разовая
   сборка в devShell), воспроизводимость для CI добавляется позже вариантом C или B
   (варианты — в списке ниже).
9. **Конвертер ONNX** (`tools/convert_to_onnx.py`) пишем и доводим до зелёных тестов
   **до Фазы 2** (Task 0.8) — иначе Task 2.4 нечего принимать.
10. **ONNX-репозиторий** — `third_party/madmom-models-onnx/models` (§2.1).
11. **Мёртвая запись** `third_party/madmom_models` в `.gitmodules` удалена (коммит
    `019e1bf`); модели приходят из вложенного сабмодуля `madmom/models`.
12. **`madmom-eval`** — отложен, вне первого релиза; CLI первого релиза — 24 программы.

**Осталось решить:**

1. **Герметичность для CI** (R16a): вариант D уже принят — подключаем C (вендоренные
   бинарники из NGC-образа) или B (nix-деривация) в тот момент, когда понадобится сборка
   на чистой машине. Триггер и критерий выбора — в описании вариантов выше.
2. **Герметичность сборки Triton** (R16a). Варианты (нужен выбор владельца):

   - **A. Разовая сборка в devShell** (что уже в Task 0.7): сеть разрешена, артефакт в
     gitignored `third_party/triton-runtime`. Дёшево и быстро, но не воспроизводимо:
     CI на чистой машине не соберёт без сети и десятков минут.
   - **B. Nix-деривация с предвыборкой всех входов**: `core`+`common`+`third_party`
     (`fetchSubmodules`) и ORT-тарболл как fixed-output в store, а сетевые `git clone`
     внутри ExternalProject подавляются (`git config url.<store>.insteadOf`, пре-заполнение
     `TRITON_THIRD_PARTY_INSTALL_PREFIX`). Воспроизводимо, но это написание деривации
     с нуля (дни) и хрупко на каждом бампе версии Triton.
   - **C. Вендоренный бинарный артефакт из NGC-образа** `nvcr.io/nvidia/tritonserver:24.08-py3`:
     один раз (там, где есть docker) вынуть `libtritonserver.so`, `libtriton_onnxruntime.so`,
     ORT-либы и `backends/`, положить в отдельный артефакт/релиз с хешами. Ноль сборки
     в CI, ровно те бинари, что у эталонного сервера; минусы — размер (сотни МБ) и условия
     распространения бинарников NVIDIA (юридический вопрос, надо проверять).
   - **D. Гибрид (рекомендую): A сейчас + C или B позже.** Основной путь — сборка из
     исходников (как решено), а воспроизводимость для CI добавляется отдельной задачей,
     когда/если CI понадобится. Критерий выбора между C и B: допустимо ли тащить
     бинарники NVIDIA (C) и готов ли ты вложить дни в деривацию (B).
3. **Размер/время сборки:** `core` с третьими сторонами (gRPC, protobuf, prometheus)
   — это десятки минут и гигабайты. Если хочется радикально меньше — можно отдельной
   задачей поднять конфигурацию без gRPC/HTTP-эндпоинтов (in-process они не нужны),
   но это правка чужого CMake (спрошу отдельно, сейчас не трогаю).
