# Target Tracker for Jetson Orin NX

Минимальный прототип системы сопровождения цели на NVIDIA Jetson Orin NX.

## Возможности первого этапа

- открытие XIMEA-камеры через `ximea.xiapi`;
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
python3 main.py --tracker CSRT
python3 main.py --tracker KCF --exposure-us 5000 --gain-db 3
```

Управление:

- `s` - выбрать область цели;
- `r` - сбросить текущую цель;
- `q` или `Esc` - выйти.

## Структура

```text
src/
├── app.py
├── cameras/
│   ├── base.py
│   └── ximea_camera.py
├── common/
│   └── frame.py
├── selection/
│   └── roi_selector.py
├── tracking/
│   └── opencv_tracker.py
└── ui/
    └── opencv_display.py
```

Код XIMEA изолирован в `src/cameras/ximea_camera.py`. Остальная система работает только через общий интерфейс `CameraSource`.
