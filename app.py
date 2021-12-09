import os
import cv2
import uuid
import json
import imutils
import numpy as np
from fpdf import FPDF
from PIL import Image
from werkzeug.utils import secure_filename
from skimage.filters import threshold_local
from flask import Flask, request, Response, redirect, render_template, abort, send_from_directory, url_for


# Configure application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024*1024*2
app.config['UPLOAD_EXTENSIONS'] = ['.jpg']
# app.config['UPLOAD_PATH'] = 'uploads'
app.config['UPLOAD_APP'] = 'app'

if os.path.isdir('pdf') is False:
    os.mkdir('pdf')
if os.path.isdir('uploads') is False:
    os.mkdir('uploads')


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def ImageProcess(image):
    path_file = ('static/images/%s.jpg'%uuid.uuid4().hex)
    
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # show the original image and the edge detected image
    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 95, offset = 10, method = "gaussian") # 11 , 10
    warped = (warped > T).astype("uint8") * 255
    # show the original and scanned images
    # print("STEP 3: Apply perspective transform")
    # cv2.imshow("Original", imutils.resize(orig, height = 800))
    # cv2.imshow("Scanned", imutils.resize(warped, height = 800))
    # cv2.waitKey(0)

    # perspective = cv2.getPerspectiveTransform(contour_approximation, points)
    # wrap = cv2.warpPerspective(gray_image, perspective, (800, 1064))

    cv2.imwrite(path_file, warped)
    # print(path_file)
    return json.dumps(path_file)



@app.route("/")
def index():
    pdf_path = os.path.join('pdf')
    for f in os.listdir(pdf_path):
        os.remove(os.path.join(pdf_path, f))
    return render_template("index.html")

@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "POST":

        uploaded_file = request.files.get('file')
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            path = os.path.join('uploads', filename)
            uploaded_file.save(path)
        return redirect("/")

    else:

        # TODO: Display the entries in the database on index.html
        files = os.listdir('pdf')
        return render_template("demo.html", upfiles=files, le=len(files))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('pdf', filename, as_attachment=True)

def makePdf(pdfFileName, listPages, dir = ''):
    if (dir):
        dir += "/"

    cover = Image.open(dir + str(listPages[0]) + ".jpg")
    width, height = cover.size

    pdf = FPDF(unit = "pt", format = [width, height])

    for page in listPages:
        pdf.add_page()
        pdf.image(dir + str(page) + ".jpg", 0, 0)

    pdf.output("pdf/" + pdfFileName + ".pdf", "F")

@app.route("/convert", methods=['POST'])
def convert():
    path = os.path.join('uploads')
    pdf_path = os.path.join('pdf')
    processed = os.path.join('static', 'images')
    image_list = os.listdir(path)
    if len(image_list) == 0:
        return redirect("/demo")
    list_pages = []
    # for image in image_list:
    #     info = image.split('.')
    #     if info[1] == 'jpg':
    #         list_pages.append(info[0])
    # print(list_pages)
    # quit()
    # print("processed", image_list)

    for image in image_list:
        info = image.split('.')
        if info[1] == 'jpg':
            full_image_path = os.path.join('uploads', image)
            # print(full_image_path)
            img = cv2.imread(full_image_path)
            # print(type(img))
            ImageProcess(img)
    # print("*", os.listdir(processed))
    processed_list = os.listdir(processed)
    for image in processed_list:
        info = image.split('.')
        if info[1] == 'jpg':
            list_pages.append(info[0])

    file_name = str(uuid.uuid4().hex)
    for f in os.listdir(pdf_path):
        os.remove(os.path.join(pdf_path, f))
    # print("##", list_pages)
    makePdf(file_name, list_pages, 'static/images')
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    for f in os.listdir(processed):
        os.remove(os.path.join(processed, f))
    return redirect("/demo")


@app.errorhandler(413)
def too_large(e):
    return "File is too large for use", 413



@app.route("/api/upload", methods=['POST'])
def upload():
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img_processed = ImageProcess(img)
    # print(img_processed)
    return Response(response=img_processed, status=200, mimetype="application/json")