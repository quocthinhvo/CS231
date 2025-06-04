<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>NHẬP MÔN THỊ GIÁC MÁY TÍNH</b></h>

## THÀNH VIÊN NHÓM
| STT    | MSSV          | Họ và Tên              |Chức Vụ    | Github                                                  | Email                   |
| ------ |:-------------:| ----------------------:|----------:|--------------------------------------------------------:|-------------------------:
| 1      | 22520006 | Võ Quốc Thịnh        |Nhóm trưởng|[quocthinhvo](https://github.com/quocthinhvo)  |22520006@gm.uit.edu.vn   |

## GIỚI THIỆU MÔN HỌC
* **Tên môn học:** Nhập môn thị giác máy tính
* **Mã môn học:** CS231.P21
* **Năm học:** HK2 (2024 - 2025)
* **Giảng viên**: TS. Mai Tiến Dũng

## CÀI ĐẶT

### Tải dataset

Dataset được sử dụng trong đồ án là DeepFashion2, có thể tải về tại [đây](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes)

### Data process

Do dataset quá lớn, nên chúng ta chỉ lấy 50.000 ảnh ngẫu nhiên để huấn luyện. Trong đó 40.000 ảnh được dùng để huấn luyện và 10.000 ảnh còn lại được dùng để val.

Thứ tự các bước thực hiện như sau:

1. Chuyển đổi DeepFashion2 gốc sang Yolo trong file `data_process/convert2yolo.py`

2. Đổi format ground truth sang Yolo trong file `data_process/convert2bbox.py`

3. Đổi dữ liệu sang định dạng để train CNN (Resnet) trong file `data_process/convert2cnn.py`