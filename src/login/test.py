import time
import requests
import json
import datetime


class WeChat:
    def __init__(self):
        self.CORPID = 'wwe977300792ec127b'  # 企业ID，在管理后台获取
        self.CORPSECRET = 'hauIpGrKIQNlt_6GB7eMNX2zoqJdt_5ukJhPhdh5NZc'  # 自建应用的Secret，每个自建应用里都有单独的secret
        self.AGENTID = '1000002'  # 应用ID，在后台应用中获取
        self.TOUSER = "YuChunXiang"  # 接收者用户名,多个用户用|分割

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
            with open('../tmp/access_token.conf', 'w') as f:
                access_token = self._get_access_token()
                cur_time = time.time()
                f.write('\t'.join([str(cur_time), access_token]))
                return access_token
        else:
            cur_time = time.time()
            if 0 < cur_time - float(t) < 7260:
                return access_token
            else:
                with open('../tmp/access_token.conf', 'w') as f:
                    access_token = self._get_access_token()
                    f.write('\t'.join([str(cur_time), access_token]))
                    return access_token

    def send_data(self, message):
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

    def send_msg_txt_img(self, name):
        headers = {"Content-Type": "text/plain"}
        send_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534'
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime("%Y-%m-%d-%H:%M:%S")
        send_data = {
            "msgtype": "news",  # 消息类型，此时固定为news
            "news": {
                "articles": [  # 图文消息，一个图文消息支持1到8条图文
                    {
                        "title": "检测到人脸，身份为 " + name,  # 标题，不超过128个字节，超过会自动截断
                        "description": "现在是" + time_str + "\n存在于人脸库中，允许通过",  # 描述，不超过512个字节，超过会自动截断
                        "url": "www.baidu.com",  # 点击后跳转的链接。
                        "picurl": "./cache/2021-04-14-15-50-45.jpg"
                        # 图文消息的图片链接，支持JPG、PNG格式，较好的效果为大图 1068*455，小图150*150。
                    },
                    # {
                    #     "title": "我的CSDN - 魏风物语",  # 标题，不超过128个字节，超过会自动截断
                    #     "description": "坚持每天写一点点",  # 描述，不超过512个字节，超过会自动截断
                    #     "url": "https://blog.csdn.net/itanping",  # 点击后跳转的链接。
                    #     "picurl": "http://res.mail.qq.com/node/ww/wwopenmng/images/independent/doc/test_pic_msg1.png"
                    #     # 图文消息的图片链接，支持JPG、PNG格式，较好的效果为大图 1068*455，小图150*150。
                    # }
                ]
            }
        }

        res = requests.post(url=send_url, headers=headers, json=send_data)
        print(res.text)



if __name__ == '__main__':
    wx = WeChat()
    wx.send_msg_txt_img('cyu')
    # wx.send_data("这是程序发送的第1条消息！\n Python程序调用企业微信API,从自建应用“告警测试应用”发送给管理员的消息！")
    # wx.send_data("这是程序发送的第2条消息！")
    # wx.send_data("这是程序发送的第2条消息！")

