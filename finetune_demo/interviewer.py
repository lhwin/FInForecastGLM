import pdfplumber
import os
def pdf_to_text(pdf_path, txt_path):
    # 打开PDF文件
    with pdfplumber.open(pdf_path) as pdf:
        # 创建一个空的字符串，用于存储转换后的文本
        text = ''
        # 遍历PDF中的每一页
        for page in pdf.pages:
            # 提取文本
            text += page.extract_text()

    # 将文本写入到TXT文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

def interviewer(txt_path, start=1, group_length=5):
    with open(txt_path, 'r', encoding='utf-8') as f:
         text = f.read()
    questions = text.split("*\n")
    length = len(questions)
    index = start-1
    while index < length:

        tmp_question = questions[index:index+group_length]
        print("目前进行 第{}至{} 共{}题".format(index, index+group_length, length))
        for i, q in enumerate(tmp_question):
            content = q.split("\n")
            qs = content[0]
            answer = content[1:]
            print("第 {} 题".format(index+i))
            print(qs)

            sure = input("Do you know?")

            if sure == "n":
                print("第 {} 题答案：".format(index))
                print("\n".join(answer))

            if i>=group_length-1:
                next_group = input("do you know next group?")
                if next_group.lower() == "y":
                    index += group_length
                    break
                else:
                    break
        os.system("cls")



if __name__ == "__main__":
    pdf_file_path = './datas/大语言模型面试题最终版.pdf'
    txt_file_path = './datas/大语言模型面试题最终版.txt'
    # pdf_to_text(pdf_file_path, txt_file_path)
    interviewer(txt_file_path, 94)