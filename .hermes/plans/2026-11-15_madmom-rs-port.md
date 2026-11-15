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

- [ ] Шаг 1 (тест): `FramedSignal` на 9-сэмпловом сигнале: `num_frames` для
      `end='normal'`/`'extend'`, `origin='online'`/`'offline'`/`'stream'`,
      `frame_rate`, `overlap_factor`, срез `[2:5]` даёт новый `FramedSignal`
      со сдвинутым origin.
- [ ] Шаг 2 (тест): `signal_frame` на границах: `start<0`, `stop>len`,
      `pad='repeat'` (повтор первого/последнего сэмпла) и `pad=0`.
- [ ] Шаг 3 (реализация): `signal_frame` — точный перенос формулы
      `start = index*hop - frame_size/2 - origin`.
- [ ] Шаг 4 (тест): `smooth` (kernel int → окно Хэмминга, `np.convolve 'same'`;
      2-D → `convolve2d 'same'`), `normalize` (int16 → /32767), `remix`
      (моно=среднее, `channel=N` — выбор канала), `rescale`, `trim`, `energy`,
      `rms`, `spl`.
- [ ] Шаг 5: перенести doctest-числа из `signal.py` (41 пример) как тесты.
- [ ] Шаг 6: коммит `feat(audio): signal, framing and signal functions`.

#### Task 1.2: `io::audio` (WAV + ffmpeg)

**Files:** Create `crates/madmom-io/src/audio.rs`; Test `crates/madmom-io/tests/audio.rs`
(порт `tests/test_io_audio.py`, 27 тестов).

- [ ] Шаг 1 (тест): `load_wave_file` для `sample.wav`, `sample_22050.wav`,
      `stereo_sample.wav` (int16, num_channels, sample_rate); ошибка
      `LoadAudioFileError` для `foo_bar.wav`; `write_wave_file` round-trip.
- [ ] Шаг 2 (тест): `load_audio_file` dispatch: `.wav` без ресемплинга → свой
      ридер; `.m4a`/`.flac`/нужен ресемплинг → ffmpeg; `start`/`stop`/`channel`/`gain`/`norm`.
- [ ] Шаг 3 (тест): ReplayGain (`stereo_sample_rg.flac`, `stereo_chirp_rg.flac`) —
      ожидаемые значения из `tests/test_io_audio.py`.
- [ ] Шаг 4 (реализация): `_ffmpeg_call` — аргументы в том же порядке
      (`-v quiet -y`, `-ss`, `-f`, `-t`, `-ac`, `-ar`, `-af pan=mono|c0=cN`).
- [ ] Шаг 5: коммит `feat(io): wav + ffmpeg audio loading`.

#### Task 1.3: `stft`

**Files:** Create `crates/madmom-audio/src/stft.rs`; Test `…/tests/stft.rs`
(порт `tests/test_audio_stft.py`, 27 тестов).

**Interfaces (Produces):** `stft()`, `fft_frequencies()`, `phase()`,
`local_group_delay()`, `ShortTimeFourierTransform` (+ `Processor`),
`Phase`, `LocalGroupDelay`, `ShortTimeFourierTransformProcessor`.

- [ ] Шаг 1 (тест): doctest-значения `fft_frequencies(4, 44100)`;
      `stft` без окна/с окном Ханна, `circular_shift=True` (свап половин
      с дополнением нулями при `fft_size > frame_size`), `include_nyquist`.
- [ ] Шаг 2 (реализация): масштабирование окна `window / iinfo(dtype).max`
      при целочисленном сигнале (D5) — иначе паритет невозможен.
- [ ] Шаг 3 (тест): сверка `stft` с `tests/golden/stft_2048.npy` (`≤1e-5`).
- [ ] Шаг 4: коммит `feat(audio): short-time Fourier transform`.

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

- [ ] Шаг 1 (тест): все скалярные преобразования частот (doctest-числа).
- [ ] Шаг 2 (тест): матрицы `MelFilterbank(40 bands, 20..17000)`,
      `LogarithmicFilterbank(12/octave, num_bands=6/12/24, 30..17000, norm=True)`;
      `frequencies2bins(unique=True)` — «ближайший бин» с округлением
      (`indices -= freqs - left < right - freqs`).
- [ ] Шаг 3: `SemitoneBandpassFilterbank` (`ellip`+`filtfilt`) — реализуется
      последней в задаче, допускается отдельный коммит и `#[cfg]`-флаг, если
      эллиптический фильтр ещё не готов (тест не удалять).
- [ ] Шаг 4: коммит `feat(audio): filterbanks`.

#### Task 1.5: `spectrogram` (Spectrogram/Filtered/Log/Diff/MultiBand)

**Files:** Create `crates/madmom-audio/src/spectrogram.rs`;
Test `…/tests/spectrogram.rs` (порт `tests/test_audio_spectrogram.py`, 49 тестов).

- [ ] Шаг 1 (тест): `Spec` = |STFT|; `FilteredSpectrogram` = `spec · filterbank`;
      `LogarithmicSpectrogram` (`log10`, `mul`, `add`); `SpectrogramDifference`
      (`diff_ratio`, `diff_frames`, `diff_max_bins`, `positive_diffs`, `stack_diffs`);
      `SuperFluxProcessor` (композиция).
- [ ] Шаг 2 (тест): сверка с `tests/golden/{filt_6_30_17000,logspec,diff_0.5_pos}.npy`.
- [ ] Шаг 3 (тест): `MultiBandSpectrogram`, `SemitoneBandpassSpectrogram` —
      на известных из тестов madmom значениях.
- [ ] Шаг 4: коммит `feat(audio): spectrograms and differences`.

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
- [ ] Шаг 2: `start_server`: `Options::new(repo)`, `.backend_directory(...)`,
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

#### Task 3.1: `ml::hmm`

**Files:** Create `crates/madmom-ml/src/hmm.rs`; Test `…/tests/hmm.rs`
(порт `tests/test_ml_hmm.py`, 9 тестов + doctest).

**Interfaces (Produces):** `TransitionModel { states, pointers, probabilities }`,
`make_sparse/make_dense/from_dense`, `ObservationModel` (trait),
`DiscreteObservationModel`, `HiddenMarkovModel::{viterbi, forward, forward_step, reset}`.

- [ ] Шаг 1 (тест): doctest `TransitionModel.from_dense([0,1,0,1],[0,0,1,1],[0.7,0.3,0.6,0.4])`
      + `DiscreteObservationModel([[0.2,0.3,0.5],[0.7,0.1,0.2]])` → `viterbi([0,0,1,1,0,0,0,2,2])`
      = `[1,1,0,0,1,1,1,0,0]`, `log_p ≈ -12.87`; таблица `forward` из докстринга.
- [ ] Шаг 2 (тест): `make_sparse` поднимает `ValueError`, если распределение не
      суммируется в 1; CSR-инвариант: `pointers[state]..pointers[state+1]` покрывает
      все переходы *в* `state`.
- [ ] Шаг 3 (тест): Viterbi с `-inf` → предупреждение и пустой путь (`is_empty`).
- [ ] Шаг 4: коммит `feat(ml): HMM viterbi/forward core`.

#### Task 3.2: Пространства состояний и модели наблюдений биений

**Files:** Create `crates/madmom-features/src/beats_hmm.rs`;
Test `…/tests/beats_hmm.rs` (порт `tests/test_features_beats_hmm.py`, 17 тестов).

**Interfaces (Produces):** `BeatStateSpace`, `BarStateSpace`,
`MultiPatternStateSpace`, `BeatTransitionModel`, `BarTransitionModel`,
`MultiPatternTransitionModel`, `RNNBeatTrackingObservationModel`,
`RNNDownBeatTrackingObservationModel`, `GMMPatternTrackingObservationModel`,
`exponential_transition`.

- [ ] Шаг 1 (тест): `BeatStateSpace(40, 400)` — `num_states`, `num_intervals`,
      `first_states`, `last_states`, лог-шкала при `num_intervals < len(linear)`.
- [ ] Шаг 2 (тест): `exponential_transition` — `np.spacing(1)`-порог,
      нормировка по строкам, `transition_lambda=None` → диагональная матрица.
- [ ] Шаг 3 (тест): `BarStateSpace(4, …)` + `MultiPatternStateSpace` — offsets
      `first_states`/`last_states` совпадают с эталоном.
- [ ] Шаг 4: коммит `feat(features): beat/downbeat HMM state and observation models`.

#### Task 3.3: Пик-пикинг онсетов

**Files:** Create `crates/madmom-features/src/onsets.rs` (частично: `peak_picking`,
`OnsetPeakPickingProcessor`, 11 ODF-функций);
Test `…/tests/onsets.rs` (порт `tests/test_features_onsets.py`, 31 тест).

- [ ] Шаг 1 (тест): ODF-числа из тестов madmom: `high_frequency_content`,
      `spectral_diff`, `spectral_flux`, `superflux`, `complex_flux`,
      `modified_kullback_leibler`, `phase_deviation`, `weighted_phase_deviation`,
      `normalized_weighted_phase_deviation`, `complex_domain`,
      `rectified_complex_domain` (первые 6 значений на `sample_log_filt_spec`).
- [ ] Шаг 2 (тест): `peak_picking` с `pre_avg/post_avg/pre_max/post_max`
      (скользящие mean/max с `origin`), `threshold`, `smooth`.
- [ ] Шаг 3 (тест): `OnsetPeakPickingProcessor` offline (`combine=0.03`,
      `delay`, `fps=100`) и online (буфер, без будущей информации); проверка
      `process_forward == process_online`.
- [ ] Шаг 4: коммит `feat(features): onset detection functions and peak picking`.

#### Task 3.4: DBN- и CRF-трекеры биений

**Files:** Create `crates/madmom-features/src/beats.rs`, `…/beats_crf.rs`;
Test `…/tests/beats.rs` (порт `tests/test_features_beats.py`, 11 тестов).

**Константы (обязательны к точному переносу):** `DBNBeatTrackingProcessor`:
`MIN_BPM=55`, `MAX_BPM=215`, `NUM_TEMPI=None`, `TRANSITION_LAMBDA=100`,
`OBSERVATION_LAMBDA=16`, `THRESHOLD=0`, `CORRECT=True`;
`BeatTrackingProcessor`: `LOOK_ASIDE=0.2`, `LOOK_AHEAD=10`;
`CRFBeatDetectionProcessor`: `INTERVAL_SIGMA=0.18`, `FACTORS=[0.5,0.67,1.0,1.5,2.0]`,
`NUM_INTERVALS=5`, `MIN_BPM=20`, `MAX_BPM=240`, `ACT_SMOOTH=0.09`, `HIST_SMOOTH=7`.

- [ ] Шаг 1 (тест): `threshold_activations` (первый/последний кадр ≥ threshold).
- [ ] Шаг 2 (тест): `DBNBeatTrackingProcessor` offline: `viterbi` → `correct=True`
      (пик активаций внутри beat-range через `np.diff(pointers[path])`) и
      `correct=False` (`argrelmin(state_positions[path], mode='wrap')`);
      online: `forward` + `argmax` + правило `next_beat = last + 60/max_bpm`.
- [ ] Шаг 3 (тест): сверка с `sample.dbn_beat_tracker.txt` (точные индексы кадров).
- [ ] Шаг 4 (тест): `BeatTrackingProcessor` (`detect_beats` c `hamming`-окном и
      рекурсией) = `sample.beat_tracker.txt`; `BeatDetectionProcessor` =
      `sample.beat_detector.txt`; `CRFBeatDetectionProcessor` = `sample.crf_beat_detector.txt`.
- [ ] Шаг 5: коммит `feat(features): beat trackers (DBN, tracking, CRF)`.

#### Task 3.5: Оценка темпа

**Files:** Create `crates/madmom-features/src/tempo.rs`;
Test `…/tests/tempo.rs` (порт `tests/test_features_tempo.py`, 37 тестов).

**Константы:** `METHOD='comb'`, `ALPHA=0.79`, `MIN_BPM=40`, `MAX_BPM=250`,
`ACT_SMOOTH=0.14`, `HIST_SMOOTH=9`, `HIST_BUFFER=10.0`, `NO_TEMPO=NaN`.

- [ ] Шаг 1 (тест): `interval_histogram_acf`, `interval_histogram_comb`
      (банк гребёнчатых фильтров, `feed_backward`), `dominant_interval`.
- [ ] Шаг 2 (тест): `detect_tempo` (в т.ч. `interpolate=True` — `interp1d('quadratic')`,
      шаг 0.01; `argrelmax(mode='wrap')`; нормализация strengths; случаи 0/1/>1 пиков).
- [ ] Шаг 3 (тест): `TempoEstimationProcessor` (`fps`, `min_bpm/max_bpm`,
      `histogram_processor`, online-режим), `TCNTempoHistogramProcessor`
      (вход — пары из TCN) = `sample.tcn_tempo_detector.txt`;
      `TempoDetector` = `sample.tempo_detector.txt`.
- [ ] Шаг 4: коммит `feat(features): tempo estimation`.

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
| R14 | Задержка online-инференса (1 кадр за вызов при 100 fps) | инференс локальный (in-process, не сеть), но вызов на кадр дорог: замер в Task 2.3 шаг 6; при провале — батчинг N кадров |
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
