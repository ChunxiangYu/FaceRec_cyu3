# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
import face
import sys
import requests
import base64
import hashlib
import datetime
import facenet
import align.detect_face
import numpy
import pickle
from scipy import misc
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import requests
import json
import imageRec

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
add_name = ''
import face_preprocess
from PyQt5.QtWidgets import QMessageBox


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # self.face_recognition = face.Recognition()
        self.face_detection = Detection()
        self.face_detection_capture = face.Detection()
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_capture = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0

        self.CORPID = 'wwe977300792ec127b'
        self.CORPSECRET = 'hauIpGrKIQNlt_6GB7eMNX2zoqJdt_5ukJhPhdh5NZc'
        self.AGENTID = '1000002'
        self.TOUSER = "YuChunXiang"  # 接收者用户名,多个用户用|分割

        modelConfiguration = "model/yolov3.cfg";
        modelWeights = "model/yolov3.weights";
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _get_access_token(self):
        url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken'
        values = {'corpid': self.CORPID,
                  'corpsecret': self.CORPSECRET,
                  }
        req = requests.post(url, params=values)
        data = json.loads(req.text)
        return data["access_token"]

    def get_access_token(self):
        try:
            with open('./tmp/access_token.conf', 'r') as f:
                t, access_token = f.read().split()
        except:
            with open('./tmp/access_token.conf', 'w') as f:
                access_token = self._get_access_token()
                cur_time = time.time()
                f.write('\t'.join([str(cur_time), access_token]))
                return access_token
        else:
            cur_time = time.time()
            if 0 < cur_time - float(t) < 7260:
                return access_token
            else:
                with open('./tmp/access_token.conf', 'w') as f:
                    access_token = self._get_access_token()
                    f.write('\t'.join([str(cur_time), access_token]))
                    return access_token

    def send_msg(self, message):
        send_url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=' + self.get_access_token()
        send_values = {
            "touser": self.TOUSER,
            "msgtype": "text",
            "agentid": self.AGENTID,
            "text": {
                "content": message
            },
            "safe": "0"
        }
        send_msges = (bytes(json.dumps(send_values), 'utf-8'))
        respone = requests.post(send_url, send_msges)
        respone = respone.json()  # 当返回的数据是json串的时候直接用.json即可将respone转换成字典
        return respone["errmsg"]

    def send_msg_txt(self, username, status):
        headers = {"Content-Type": "text/plain"}
        send_url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534"
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime("%Y-%m-%d-%H:%M:%S")
        if username == '陌生人':
            des = "\n不存在于人脸库中，禁止通过!"
        else:
            if str(status) == '0':
                des = "\n检测到佩戴口罩\n存在于人脸库中，允许通过."
            else:
                des = "\n警告，检测到未佩戴口罩\n存在于人脸库中，请佩戴口罩后通过."

        send_data = {
            "msgtype": "text",  # 消息类型，此时固定为text
            "text": {
                "content": "检测到人脸，身份为 " + username + "\n现在是" + time_str + des,  # 文本内容，最长不超过2048个字节，必须是utf8编码
                # "mentioned_list": ["@all"],
                # # userid的列表，提醒群中的指定成员(@某个成员)，@all表示提醒所有人，如果开发者获取不到userid，可以使用mentioned_mobile_list
                # "mentioned_mobile_list": ["@all"]  # 手机号列表，提醒手机号对应的群成员(@某个成员)，@all表示提醒所有人
            }
        }

        res = requests.post(url=send_url, headers=headers, json=send_data)
        print(res.text)

    def send_msg_txt_img(self, username, imagename):
        headers = {"Content-Type": "text/plain"}
        send_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534'
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime("%Y-%m-%d-%H:%M:%S")
        imagepath = './inRes/' + imagename + ".jpg"
        if username == '陌生人':
            des = "\n不存在于人脸库中，禁止通过!"
        else:
            des = "\n存在于人脸库中，允许通过。 :)"
        send_data = {
            "msgtype": "news",  # 消息类型，此时固定为news
            "news": {
                "articles": [  # 图文消息，一个图文消息支持1到8条图文
                    {
                        "title": "检测到人脸，身份为 " + username,  # 标题，不超过128个字节
                        "description": "现在是" + time_str + des,  # 描述，不超过512个字节
                        "url": "www.google.com",  # 点击后跳转的链接。
                        "picurl": imagepath
                        # 图文消息的图片链接，支持JPG、PNG格式，较好的效果为大图 1068*455，小图150*150。
                    },
                ]
            }
        }

        res = requests.post(url=send_url, headers=headers, json=send_data)
        print(res.text)

    def send_image(self, image):
        with open(image, 'rb') as file:  # 转换图片成base64格式
            data = file.read()
            encodestr = base64.b64encode(data)
            image_data = str(encodestr, 'utf-8')

        with open(image, 'rb') as file:  # 图片的MD5值
            md = hashlib.md5()
            md.update(file.read())
            image_md5 = md.hexdigest()

        url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534"  # 填上机器人Webhook地址
        headers = {"Content-Type": "application/json"}
        data = {
            "msgtype": "image",
            "image": {
                "base64": image_data,
                "md5": image_md5
            }
        }
        result = requests.post(url, headers=headers, json=data)
        return result

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.opencamera = QtWidgets.QPushButton(u'人脸识别')
        self.addface = QtWidgets.QPushButton(u'建库')
        self.captureface = QtWidgets.QPushButton(u'采集人脸')
        self.saveface = QtWidgets.QPushButton(u'保存人脸')
        self.opencamera.setMinimumHeight(50)
        self.addface.setMinimumHeight(50)
        self.captureface.setMinimumHeight(50)
        self.saveface.setMinimumHeight(50)
        self.lineEdit = QtWidgets.QLineEdit(self)  # 创建 QLineEdit
        self.lineEdit.textChanged.connect(self.text_changed)
        self.lineEdit.setMinimumHeight(50)

        # self.opencamera.move(10, 30)
        # self.captureface.move(10, 50)
        self.lineEdit.move(15, 350)

        # 信息显示
        self.showcamera = QtWidgets.QLabel()
        # self.label_move = QtWidgets.QLabel()
        self.lineEdit.setFixedSize(70, 30)

        self.showcamera.setFixedSize(641, 481)
        self.showcamera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.opencamera)
        self.__layout_fun_button.addWidget(self.addface)
        self.__layout_fun_button.addWidget(self.captureface)
        self.__layout_fun_button.addWidget(self.saveface)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.showcamera)

        self.setLayout(self.__layout_main)
        # self.label_move.raise_()
        self.setWindowTitle(u'FaceRec')

    def slot_init(self):
        self.opencamera.clicked.connect(self.button_open_camera_click)
        self.addface.clicked.connect(self.button_add_face_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera_capture.timeout.connect(self.capture_camera)
        self.captureface.clicked.connect(self.button_capture_face_click)
        self.saveface.clicked.connect(self.save_face_click)

    def text_changed(self):
        global add_name
        add_name = self.lineEdit.text()
        print(u'姓名为：%s' % add_name)

    def button_open_camera_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        self.face_recognition = face.Recognition()
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

                self.opencamera.setText(u'关闭识别')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.showcamera.clear()
            self.opencamera.setText(u'人脸识别')

    def show_camera(self):
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(self.image, (640, 480))
        # face_detection = self.face.Detection()
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        faces = self.face_recognition.identify(show)
        if faces is not None:
            if faces is not None:
                img_PIL = Image.fromarray(show)
                font = ImageFont.truetype('simsun.ttc', 40)
                # 字体颜色
                fillColor1 = (255, 0, 0)
                fillColor2 = (0, 255, 0)
                self.images = self.cap.read()
                curr_time = datetime.datetime.now()
                name_str = curr_time.strftime("%Y-%m-%d-%H-%M-%S")
                draw = ImageDraw.Draw(img_PIL)
                for face in faces:
                    face_bb = face.bounding_box.astype(int)
                    draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
                    draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
                    draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
                    draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
                    if face.name is not None:
                        if face.name == 'unknown':
                            draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                            ret, frame = self.cap.read()
                            cv2.imwrite(r'./outRes/' + name_str + ".jpg", frame)
                            imageRec.getimage(r'./outRes/' + name_str + ".jpg", r'./outRes_result/' + name_str + ".jpg")
                            status = imageRec.getclassids(r'./outRes/' + name_str + ".jpg")
                            self.send_msg_txt('陌生人', status)
                            self.send_image(
                                r"C:\\Users\\yu146\\PycharmProjects\\FaceRec_cyu3\\src\\outRes_result\\" + name_str + ".jpg")
                        else:
                            draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
                            ret, frame = self.cap.read()
                            cv2.imwrite(r'./inRes/' + name_str + ".jpg", frame)
                            imageRec.getimage(r'./inRes/' + name_str + ".jpg", r'./inRes_result/' + name_str + ".jpg")
                            status = imageRec.getclassids(r'./inRes/' + name_str + ".jpg")
                            self.send_msg_txt(face.name, status)
                            self.send_image(
                                r"C:\\Users\\yu146\\PycharmProjects\\FaceRec_cyu3\\src\\inRes_result\\" + name_str + ".jpg")

            show = numpy.asarray(img_PIL)
        # show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def button_add_face_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        model = "123456/3001w-train.pb"
        traindata_path = "../data/image"
        feature_files = []
        face_label = []
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                facenet.load_model(model)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                for images in os.listdir(traindata_path):
                    print(images)
                    filename = os.path.splitext(os.path.split(images)[1])[0]
                    image_path = traindata_path + "/" + images
                    images = self.face_detection.find_faces(image_path)
                    if images is not None:
                        face_label.append(filename)
                        # Run forward pass to calculate embeddings
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print(emb)
                        feature_files.append(emb)
                    else:
                        print('no find face')
                write_file = open('123456/knn_classifier.pkl', 'wb')
                pickle.dump(feature_files, write_file, -1)
                pickle.dump(face_label, write_file, -1)
                write_file.close()
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "建库",
                                        "建库完成",
                                        QMessageBox.Yes | QMessageBox.No)

    def button_capture_face_click(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_camera_capture.start(30)

    def capture_camera(self):
        flag, self.images = self.cap.read()
        self.images = cv2.cvtColor(self.images, cv2.COLOR_BGR2RGB)
        show_images = self.images
        faces = self.face_detection_capture.find_faces(show_images)
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(show_images,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              (0, 255, 0), 2)
        show_images = numpy.asarray(show_images)
        showImage = QtGui.QImage(show_images.data, show_images.shape[1], show_images.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def save_face_click(self):
        global add_name
        imagepath = os.sep.join(['../data/image/', add_name + '.jpg'])
        print('faceID is:', add_name)
        if add_name == '':
            reply = QMessageBox.information(self,  # 使用infomation信息框
                                            "人脸用户名",
                                            "请在文本框输入人脸的用户名",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            self.images = cv2.cvtColor(self.images, cv2.COLOR_RGB2BGR)
            cv2.imencode(add_name + '.jpg', self.images)[1].tofile(imagepath)
            # cv2.imwrite('../data/gump/' + 'cyu' + '.jpg', self.images)

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


# def add_overlays(frame, faces):
#     if faces is not None:
#         img_PIL = Image.fromarray(frame)
#         font = ImageFont.truetype('simsun.ttc', 40)
#         # 字体颜色
#         fillColor1 = (255, 0, 0)
#         fillColor2 = (0, 255, 0)
#         draw = ImageDraw.Draw(img_PIL)
#         for face in faces:
#             face_bb = face.bounding_box.astype(int)
#             draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
#             draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
#             draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
#             draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
#             if face.name is not None:
#                 if face.name == 'unknown':
#                     draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
#                 else:
#                     draw.text((face_bb[0], face_bb[1]), '你是谁', font=font, fill=fillColor1)
#         frame = numpy.asarray(img_PIL)
#         return frame


class Detection:
    minsize = 40  # minimum size of face
    threshold = [0.8, 0.9, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=112, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image_paths):
        img = misc.imread(os.path.expanduser(image_paths), mode='RGB')
        _bbox = None
        _landmark = None
        bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                               self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        img_list = []
        max_Aera = 0
        if nrof_faces > 0:
            if nrof_faces == 1:
                bindex = 0
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                # cv2.imwrite('1.jpg',warped)
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
            else:
                for i in range(nrof_faces):
                    _bbox = bounding_boxes[i, 0:4]
                    if _bbox[2] * _bbox[3] > max_Aera:
                        max_Aera = _bbox[2] * _bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    # ui.send_data("nihao1")
    sys.exit(app.exec_())
