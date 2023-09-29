import csv

label = [
    "cứu tôi với cháy to quá",
    "nhà tôi đang cháy",
    "tôi đang mắc kẹt trong đám cháy",
    "nhiều khói quá tôi không thở được",
    "tôi đang bị bỏng và mắc kẹt trong đám cháy",
    "xung quanh cháy to quá",
    "tòa chung cư tôi ở đang bị cháy",
    "có hỏa hoạn cứu tôi với",
    "tôi đang bị ngạt khói tôi không thể thở được",
    "tôi bị lạc",
    "tôi không biết mình đang ở đâu",
    "tôi không thể tìm đường về nhà",
    "tôi không biết phải đi đâu",
    "tôi đang bị lạc trong rừng",
    "tôi đang bị lạc trong một hang động",
    "tôi đang bị lạc trên một hòn đảo",
    "tôi bị kẹt trong xe",
    "tôi không thể thoát ra",
    "tôi đang bị mắc kẹt trong một chiếc thang",
    "tôi đang bị mắc kẹt trong một tòa nhà bỏ hoang",
    "tôi đang bị mắc kẹt trong một cơn bão",
    "tôi bị mắc kẹt trong một chiếc xe tôi không thể mở cửa"
]

with open('audiofolder\metadata.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    field = ["file_name", "transcription"]
    writer.writerow(field)
    for i in range (1, 45):
        writer.writerow([f"{i}.wav", label[i%22-1]])
    