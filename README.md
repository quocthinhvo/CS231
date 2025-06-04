<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>NHẬP MÔN THỊ GIÁC MÁY TÍNH</b></h>

[![DeepWiki](https://img.shields.io/badge/DeepWiki-Documentation-blue)](https://deepwiki.com/quocthinhvo/CS231)


## THÀNH VIÊN NHÓM
| STT    | MSSV          | Họ và Tên              |Chức Vụ    | Github                                                  | Email                   |
| ------ |:-------------:| ----------------------:|----------:|--------------------------------------------------------:|-------------------------:
| 1      | 22520006 | Võ Quốc Thịnh        |Nhóm trưởng|[quocthinhvo](https://github.com/quocthinhvo)  |22520006@gm.uit.edu.vn   |

## GIỚI THIỆU MÔN HỌC
* **Tên môn học:** Nhập môn thị giác máy tính
* **Mã môn học:** CS231.P21
* **Năm học:** HK2 (2024 - 2025)
* **Giảng viên**: TS. Mai Tiến Dũng

## HÌNH ẢNH TỪ DEMO

<!-- chèn ảnh -->
[![Demo YOLO](/images/yolo_demo.jpg)](#)

[![Demo ResNet50](/images/resnet_demo.jpg)](#)

## MÔ TẢ ĐỒ ÁN

Mục tiêu chính của đề tài là xây dựng một mô hình học sâu có khả năng phân loại tự động các sản phẩm thời trang dựa trên hình ảnh, hướng đến việc ứng dụng vào các nền tảng thương mại điện tử và hệ thống quản lý sản phẩm.
Cụ thể, đề tài tập trung vào các mục tiêu sau:
- Nghiên cứu và triển khai mô hình học sâu cho bài toán phân loại hình ảnh, nhằm nhận diện và phân biệt các loại sản phẩm thời trang phổ biến như áo, váy, quần,...
- Xử lý bộ dữ liệu DeepFashion2 bao gồm: chọn mẫu ngẫu nhiên từ tập dữ liệu lớn, chuyển đổi định dạng, chia tập huấn luyện và kiểm tra, cũng như thực hiện các bước tiền xử lý ảnh.
- Huấn luyện mô hình trên môi trường hạn chế tài nguyên (Google Colab), theo dõi quá trình huấn luyện và đánh giá hiệu quả thông qua các chỉ số như độ chính xác, độ mất mát và khả năng tổng quát hóa.
- Phân tích kết quả và đánh giá tiềm năng ứng dụng thực tế của mô hình trong các hệ thống tự động phân loại hình ảnh sản phẩm, phục vụ mục tiêu tối ưu hóa vận hành và nâng cao trải nghiệm người dùng.


## CÀI ĐẶT

### Tải dataset

Dataset được sử dụng trong đồ án là DeepFashion2, có thể tải về tại [đây](https://www.kaggle.com/datasets/thusharanair/deepfashion2-original-with-dataframes)

### Data process

Do dataset quá lớn, nên chúng ta chỉ lấy 50.000 ảnh ngẫu nhiên để huấn luyện. Trong đó 40.000 ảnh được dùng để huấn luyện và 10.000 ảnh còn lại được dùng để val.

Thứ tự các bước thực hiện như sau:

1. Chuyển đổi DeepFashion2 gốc sang Yolo trong file `data_process/convert2yolo.py`

2. Đổi format ground truth sang Yolo trong file `data_process/convert2bbox.py`

3. Đổi dữ liệu sang định dạng để train CNN (Resnet) trong file `data_process/convert2cnn.py`

## KẾT LUẬN

Trong đề tài này, nhóm đã nghiên cứu và triển khai bài toán phân loại sản phẩm thời trang dựa trên hình ảnh, với hai hướng tiếp cận chính: sử dụng mô hình CNN (ResNet50) cho bài toán phân loại thuần túy, và sử dụng mô hình YOLOv8 cho bài toán phát hiện và phân loại kết hợp.
Kết quả huấn luyện cho thấy mô hình ResNet50 đạt độ chính xác classification lên đến 77.29%, cho thấy khả năng học tốt các đặc trưng thị giác của từng loại sản phẩm trong tập dữ liệu DeepFashion2. Đây là một kết quả khả quan đối với một mô hình backbone cổ điển như ResNet50.

Tuy nhiên, khi xét đến khả năng ứng dụng thực tế, đặc biệt trong các hệ thống cần nhận diện sản phẩm trực tiếp từ ảnh đầu vào không được cắt sẵn, thì YOLOv8 tỏ ra vượt trội. Mô hình này không chỉ tự động phát hiện vùng chứa sản phẩm mà còn phân loại chính xác, nhờ kết hợp detection và classification trong một pipeline duy nhất. Đồng thời, YOLOv8 còn có ưu điểm về tốc độ suy luận nhanh, dễ triển khai lên các ứng dụng web hoặc thiết bị biên (edge devices).
Dù vẫn tồn tại một số hạn chế như nhầm lẫn giữa các lớp có ngoại hình gần giống hoặc hiệu suất suy giảm với ảnh phân giải thấp, mô hình YOLOv8 vẫn chứng minh được tính hiệu quả và phù hợp khi triển khai trong các hệ thống thực tế như phân loại ảnh sản phẩm, gợi ý trang phục, hay tổ chức kho ảnh thời trang.
Trong tương lai, nhóm định hướng mở rộng nghiên cứu với:
- Fine-tuning sâu hơn và huấn luyện nhiều epoch hơn.
- Kết hợp thêm thông tin ngữ cảnh như mô tả sản phẩm, thẻ metadata.
- Triển khai mô hình trong các hệ thống thực tế hoặc ứng dụng doanh nghiệp.
- Sử dụng toàn bộ tập dữ liệu DeepFashion2 khi có đủ tài nguyên.
