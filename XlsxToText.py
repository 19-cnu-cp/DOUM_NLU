import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

XLSX_FILE_NAME = '2019_10_28_Recruit.xlsx'
XLSX_FILE_PATH = os.path.join('data', 'xlsx', XLSX_FILE_NAME)

TEXT_FILE_NAME = 'raw.txt'
TEXT_FILE_PATH = os.path.join('data', 'text', TEXT_FILE_NAME)

import xlrd

def extractIntent(xlsx):
    # input: xlrd.book.Book
    # output: dictionary {string, string}
    sheet = xlsx.sheets()
    function_sheet = sheet[0] # xlrd.sheet.sheet

    max_row = function_sheet.nrows

    intent_dict = {}
    for i in range(1, max_row):
        intent_dict[function_sheet.cell_value(i, 1)] = function_sheet.cell_value(i, 4)

    return intent_dict

def extractUseSheet(xlsx):
    # input: xlrd.book.Book
    # output: dictionary {string, xlrd.sheet.sheet}
    sheet = xlsx.sheets() # list
    sheet_dict = {}

    content_on = False
    for item in sheet:
        if (item.name == '||||||||'):
            content_on = True
            continue

        if (content_on):
            sheet_dict[item.name] = item

    return sheet_dict

def isEqualFunctionAndSheet(sheet, intent):
    # input: dictionary, dictionary
    # output: void
    # validation
    # 1. function에 써 놓은 기능명과 sheet의 이름이 일치하는지
    # 2. function에 써 놓은 intent와 sheet안의 intent가 일치하는지
    # 3. function의 개수와 sheet의 개수가 일치하는지
    # intent_sorted = sorted(intent.keys())
    sheet_sorted = sorted(sheet.keys())
    intent_by_sheet = {}
    for k, v in sheet.items():
        intent_by_sheet[k] = v.cell_value(1, 3)
    if (not(sorted(intent) == sorted(intent_by_sheet))):
        print("[ERROR] function 안의 intent와 sheet안의 intent가 다름.")
        import sys; sys.exit()

    # if (len(intent_sorted) == len(sheet_sorted)):
    #     for i in range(0, len(intent_sorted)):
    #         if (intent_sorted[i] == sheet_sorted[i]):
    #             continue
    #         else:
    #             print("[ERROR] function 이름과 sheet의 이름이 다름.")
    #             import sys; sys.exit()
    
    # else:
    #     print("[ERROR] function 개수와 sheet 개수가 다름.")
    #     import sys; sys.exit()

def extractIntentAndAnnotation(sheet):
    # input: dictionary {string, xlsx.sheet.sheet}
    # output: tuple in list [(intent, annotation)]
    intentAndAnnotation = []
    for k,v in sheet.items():
        nrows = v.nrows
        for row in range(1, nrows):
            intentAndAnnotation.append((v.cell_value(1, 3), v.cell_value(row, 4))) # tuple(intent, annotation)
    
    return intentAndAnnotation

def writeText(intentAndAnnotation, TEXT_FILE_PATH):
    # input: tuple in list, string
    # output: void
    import xlwt, codecs
    with codecs.open(TEXT_FILE_PATH, 'w', encoding='UTF-8') as f:
        for item in intentAndAnnotation:
            print("{}\t{}".format(item[0], item[1]), file=f)

def convertXlsxToText(XLSX_FILE_PATH, TEXT_FILE_PATH):
    # input: string(path of excel, path of text)
    # output: void
    print("XLSX TO TEXT START")

    xlsx = xlrd.open_workbook(XLSX_FILE_PATH)
    # 엑셀 오픈
    xlsx_sheet = extractUseSheet(xlsx) # dictionary
    # ||||||| 뒤의 사용하는 시트만 빼온다. {시트명, sheet내용}
    xlsx_intent = extractIntent(xlsx) # dictionary
    # 첫 번째 시트(function)에서 {기능명,intent}를 받는다.
    isEqualFunctionAndSheet(xlsx_sheet, xlsx_intent) # validation
    # {sheet_name, sheet_intent}과 {function_name, function_intent}가 일치하는지 확인
    intentAndAnnotation = extractIntentAndAnnotation(xlsx_sheet) # tuple in list
    # sheet 안의 내용물을 [(intent, annotation)]의 형태로 가져옴.
    writeText(intentAndAnnotation, TEXT_FILE_PATH)
    # 텍스트 쓰기

    print("XLSX TO TEXT END")

if __name__ == "__main__":
    convertXlsxToText(XLSX_FILE_PATH, TEXT_FILE_PATH)