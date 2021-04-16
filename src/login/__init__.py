import requests
import base64
import hashlib


def wx_image(image):
    with open(image, 'rb') as file:  # 转换图片成base64格式
        data = file.read()
        encodestr = base64.b64encode(data)
        image_data = str(encodestr, 'utf-8')
        print(image_data)

    with open(image, 'rb') as file:  # 图片的MD5值
        md = hashlib.md5()
        md.update(file.read())
        image_md5 = md.hexdigest()
        print(image_md5)

    url = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534"  # 填上机器人Webhook地址
    headers = {"Content-Type": "application/json"}
    data = {
        "msgtype": "image", "text"
        "image": {
            "base64": image_data,
            "md5": image_md5
        },
        "text": {
              "content": "123"
        }
    }
    result = requests.post(url, headers=headers, json=data)
    return result


if __name__ == '__main__':
    wx_image(r'./cache/2021-04-15-09-44-28.jpg')

    # wx_image("C:\\Users\\yu146\\PycharmProjects\\FaceRec_cyu3\\src\\cache\\2021-04-14-15-50-45.jpg")
