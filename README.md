# Target Tracker for Jetson Orin NX

Прототип системы сопровождения цели на NVIDIA Jetson Orin NX с камерой Basler
daA1600-60um (Mono8, USB3). Архитектура модульная: захват, отображение, выбор
цели, трекинг, прогноз (Kalman), перезахват и детекция разделены и имеют
собственные интерфейсы.

## Что нового в производительности

Раньше после выбора цели CSRT на полном кадре 1280x960 проседал до ~15 FPS.
Ключевые изменения:

- **Потоковый захват** (`CameraCaptureWorker`): чтение камеры вынесено в
  отдельный поток с очередью глубины 1. Медленный трекер больше не блокирует
  захват, обрабатывается только самый свежий кадр, устаревшие кадры считаются
  `dropped`, а не копятся.
- **Динамический ROI** (`DynamicROITracker`): трекер работает в небольшом окне
  вокруг последнего положения/прогноза Kalman, а не на полном кадре. Окно
  смещается, когда цель подходит к краю. Это главный источник ускорения для
  CSRT/KCF.
- **Grayscale-конвейер**: Mono8 не конвертируется в BGR в источнике камеры.
  Трекинг, ORB и template matching работают на исходном grayscale; BGR-копия
  создаётся только в момент отрисовки. Яркость 8-битного кадра не меняется.
- **Профили качества** `quality` / `balanced` / `fast` в `config.json`.
- **Профилирование этапов** (`StageProfiler`) и режим `--benchmark`.

## Запуск

```bash
python3 main.py
```

Профиль по умолчанию задаётся ключом `profile` в `config.json` (`balanced`).
Сменить профиль:

```bash
python3 main.py --profile quality
python3 main.py --profile fast
```

CLI оставлен только как необязательный override поверх конфига, например:

```bash
python3 main.py --tracker KCF --tracking-scale 0.5 --no-roi
python3 main.py --roi-scale 3.0 --reacquire-cooldown 10
```

Другой конфиг:

```bash
python3 main.py --config config-quality.json
```

Управление:

- выделение мышью — выбрать область цели;
- `r` — полный сброс цели (трекер, Kalman и память объекта очищаются);
- `q` или `Esc` — выйти.

Состояния трекинга на экране: `TRACKING` (подтверждено), `PREDICTING` (короткая
потеря, ведёт Kalman), `REACQUIRING` (идёт локальный/глобальный поиск),
`TARGET STALE` (давно не подтверждалась).

## Benchmark

Безоконный режим для сравнения «до/после». Несколько секунд без трекинга, затем
несколько секунд с трекингом (если задан ROI):

```bash
python3 main.py --benchmark
python3 main.py --benchmark --benchmark-roi 600,450,80,80
```

Выводит capture FPS, processing FPS, latency (p50/p95), dropped frames и p50/p95
по каждому этапу конвейера.

## Диагностика GPU на Jetson

```bash
python3 tools/jetson_diagnostics.py
```

Определяет модель платы (`/proc/device-tree/model`), сборку OpenCV с CUDA,
количество CUDA-устройств, наличие TensorRT/pyCUDA и записывает отчёт в
`docs/JETSON_DIAGNOSTICS.md`. Подробности и что реально можно перенести на GPU —
в `docs/JETSON_GPU.md`. Важно: CSRT/KCF в OpenCV работают на CPU; CUDA их не
ускоряет.

## Тесты

```bash
python3 tests/run_tests.py     # без зависимостей
# либо, если установлен pytest:
python3 -m pytest tests/
```

## Basler Setup

На Jetson должен быть установлен Basler pylon Software Suite для Linux ARM64.
Для USB3-камер:

```bash
sudo /opt/pylon/share/pylon/setup-usb.sh
```

После этого переподключить камеру или перезагрузить Jetson.

Проверка камеры:

```bash
python3 - <<'PY'
from pypylon import pylon

devices = pylon.TlFactory.GetInstance().EnumerateDevices()
print("devices:", len(devices))
for d in devices:
    print(d.GetModelName(), d.GetSerialNumber(), d.GetDeviceClass())
PY
```

## Структура

```text
src/
├── app.py                      # координация конвейера
├── benchmark.py                # безоконный бенчмарк
├── cameras/
│   ├── base.py                 # интерфейс CameraSource
│   ├── basler_camera.py        # реализация Basler/pypylon (изолирована)
│   └── capture_worker.py       # потоковый захват, очередь глубины 1
├── common/
│   ├── frame.py                # унифицированный кадр
│   └── profiling.py            # StageProfiler (perf_counter, p50/p95)
├── detection/
│   ├── base.py                 # интерфейс Detector + Detection
│   ├── null_detector.py        # заглушка (выключена)
│   └── tensorrt_detector.py    # точка интеграции TensorRT (стаб)
├── selection/
│   └── roi_selector.py
├── tracking/
│   ├── opencv_tracker.py       # обёртка CSRT/KCF/MOSSE/MIL
│   ├── roi_tracker.py          # DynamicROITracker
│   ├── target_state.py         # Kalman + машина состояний
│   ├── object_memory.py        # стабильный + адаптивные шаблоны, ORB
│   ├── template_reacquirer.py  # локальный/глобальный перезахват
│   └── states.py               # константы состояний
└── ui/
    └── opencv_display.py
tools/
└── jetson_diagnostics.py
docs/
└── JETSON_GPU.md
```

Код Basler/pypylon изолирован в `src/cameras/basler_camera.py`. Остальная
система работает только через интерфейсы `CameraSource` и `Frame`.
