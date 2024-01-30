import cv2
import pytesseract
import re

def extract_text(image):
    # 이미지 읽기
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지 to gray
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # 가우시안 블러 -> thresholding

    # Show gray and blurred images
    # cv2.imshow('Gray Image', gray_image)
    # cv2.imshow('Blurred Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # contours를 원본 이미지에 그리기
    # 데이터 취합
    extracted_data = pytesseract.image_to_string(threshold_image, lang='kor_k', config='--psm 6')  # tesseract
    return extracted_data, threshold_image


def find_and_extract_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        plate_roi = image[y:y+h, x:x+w]
        extracted_text, threshold_img = extract_text(plate_roi)
        # 정규 표현식을 사용하여 숫자와 한글만 추출
        extracted_text = re.sub(r'[^가-힣0-9]', '', extracted_text)
        digit_count = len([c for c in extracted_text if c.isdigit()])
        hangul_count = len([c for c in extracted_text if '가' <= c <= '힣'])
        if (digit_count == 6 or digit_count == 7) and hangul_count == 1:
            cv2.imshow('Detected Plate', plate_roi)
            cv2.imshow('Threshold Image', threshold_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return extracted_text, threshold_img

    return None, None

image = cv2.imread('isak4.jpg')
extracted_text = ""
threshold_img = None

while True:
    if image.size < 1000000:  # 작은 이미지일 경우
        extracted_text, threshold_img = find_and_extract_plate(image)
        if extracted_text is not None:
            break
        image = cv2.resize(image, (int(image.shape[1] * 1.5), int(image.shape[0] * 1.5)))
    else:  # 큰 이미지일 경우
        extracted_text, threshold_img = find_and_extract_plate(image)
        if extracted_text is not None:
            break
        image = cv2.resize(image, (int(image.shape[1] * 1.04), int(image.shape[0] * 1.04)))

print(extracted_text)
