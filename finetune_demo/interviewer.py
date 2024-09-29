import pdfplumber

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

if __name__ == "__main__":
    pdf_file_path = './datas/大语言模型面试题最终版.pdf'
    txt_file_path = './datas/大语言模型面试题最终版.txt'
    pdf_to_text(pdf_file_path, txt_file_path)