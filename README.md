# uav_tracker

python infer.py --source test_videos/chernigiv.mp4 

python track.py --trackers avtrack --source test_videos/chernigiv.mp4 


python eval_detect.py --ai 1 --conf 0.45 --modality both --max-seqs 1

--ai — які моделі ганяти (1-7 через кому, або all). Дефолт: all.
--stride — кожен N-й кадр (10 = швидко для підбору, 1 = фінальний замір). Дефолт: 10.
--conf — поріг впевненості (нижче = більше боксів, вищий recall, більше FP). Обов'язковий, дефолту немає.
--modality — яке відео брати: visible, infrared чи both. Дефолт: visible.

--iou — NMS-поріг злиття дубль-боксів. Дефолт: 0.45.
--imgsz — розмір входу моделі: 960 точніше на дрібних дронах, але повільніше за 640. Дефолт: 640.
--max-seqs — обмежити кількість сцен (для швидкої перевірки). Дефолт: 0 (всі).
--device — cuda:0/cpu. Дефолт: авто (cuda якщо є, інакше cpu).
--out — куди дописати csv. Дефолт: runs/eval_detect.csv.
--val — папка з валідацією. Дефолт: val.
