# Target Tracker for Jetson Orin NX

Минимальный прототип системы сопровождения цели на NVIDIA Jetson Orin NX.

## Возможности первого этапа

- открытие Basler daA1600-60um через `pypylon`;
- получение кадров и приведение их к формату OpenCV;
- отображение видеопотока;
- выбор цели мышью;
- инициализация OpenCV-трекера;
- отрисовка рамки цели;
- вывод координат центра цели;
- корректное закрытие камеры и окон OpenCV.

## Запуск

```bash
python3 main.py
```

Дополнительные параметры:

```bash
python3 main.py --tracker CSRT --tracking-scale 0.75
python3 main.py --tracker CSRT --tracking-scale 1.0 --serial 24802261 --exposure-us 5000 --gain-db 3
```

Для более быстрого трекинга можно переключиться на KCF и уменьшить внутренний размер кадра:

```bash
python3 main.py --tracker KCF --tracking-scale 0.5
```

Сглаживание, прогноз и перезахват:

```bash
python3 main.py --tracker CSRT --tracking-scale 0.75 --smooth-alpha 0.3 --max-lost-frames 120 --reacquire-score 0.62
```

Глобальный перезахват по запомненному объекту после ухода камеры в сторону:

```bash
python3 main.py --tracker CSRT --tracking-scale 0.75 --global-reacquire-after 8 --global-reacquire-interval 5
```

Управление:

- выделение мышью - выбрать область цели;
- `r` - сбросить текущую цель;
- `q` или `Esc` - выйти.

## Basler Setup

На Jetson должен быть установлен Basler pylon Software Suite для Linux ARM64.
Для USB3-камер нужно выполнить:

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
├── app.py
├── cameras/
│   ├── base.py
│   └── basler_camera.py
├── common/
│   └── frame.py
├── selection/
│   └── roi_selector.py
├── tracking/
│   └── opencv_tracker.py
└── ui/
    └── opencv_display.py
```

Код Basler/pypylon изолирован в `src/cameras/basler_camera.py`. Остальная система работает только через общий интерфейс `CameraSource`.
