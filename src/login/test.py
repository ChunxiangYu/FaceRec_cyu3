import time
import requests
import json
import datetime


class WeChat:
    def __init__(self):
        self.CORPID = 'wwe977300792ec127b'  # ��ҵID���ڹ����̨��ȡ
        self.CORPSECRET = 'hauIpGrKIQNlt_6GB7eMNX2zoqJdt_5ukJhPhdh5NZc'  # �Խ�Ӧ�õ�Secret��ÿ���Խ�Ӧ���ﶼ�е�����secret
        self.AGENTID = '1000002'  # Ӧ��ID���ں�̨Ӧ���л�ȡ
        self.TOUSER = "YuChunXiang"  # �������û���,����û���|�ָ�

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
        respone = respone.json()  # �����ص�������json����ʱ��ֱ����.json���ɽ�responeת�����ֵ�
        return respone["errmsg"]

    def send_msg_txt_img(self, name):
        headers = {"Content-Type": "text/plain"}
        send_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a32f71c-696a-410b-aba6-297a093fb534'
        curr_time = datetime.datetime.now()
        time_str = curr_time.strftime("%Y-%m-%d-%H:%M:%S")
        send_data = {
            "msgtype": "news",  # ��Ϣ���ͣ���ʱ�̶�Ϊnews
            "news": {
                "articles": [  # ͼ����Ϣ��һ��ͼ����Ϣ֧��1��8��ͼ��
                    {
                        "title": "��⵽���������Ϊ " + name,  # ���⣬������128���ֽڣ��������Զ��ض�
                        "description": "������" + time_str + "\n�������������У�����ͨ��",  # ������������512���ֽڣ��������Զ��ض�
                        "url": "www.baidu.com",  # �������ת�����ӡ�
                        "picurl": "./cache/2021-04-14-15-50-45.jpg"
                        # ͼ����Ϣ��ͼƬ���ӣ�֧��JPG��PNG��ʽ���Ϻõ�Ч��Ϊ��ͼ 1068*455��Сͼ150*150��
                    },
                    # {
                    #     "title": "�ҵ�CSDN - κ������",  # ���⣬������128���ֽڣ��������Զ��ض�
                    #     "description": "���ÿ��дһ���",  # ������������512���ֽڣ��������Զ��ض�
                    #     "url": "https://blog.csdn.net/itanping",  # �������ת�����ӡ�
                    #     "picurl": "http://res.mail.qq.com/node/ww/wwopenmng/images/independent/doc/test_pic_msg1.png"
                    #     # ͼ����Ϣ��ͼƬ���ӣ�֧��JPG��PNG��ʽ���Ϻõ�Ч��Ϊ��ͼ 1068*455��Сͼ150*150��
                    # }
                ]
            }
        }

        res = requests.post(url=send_url, headers=headers, json=send_data)
        print(res.text)



if __name__ == '__main__':
    wx = WeChat()
    wx.send_msg_txt_img('cyu')
    # wx.send_data("���ǳ����͵ĵ�1����Ϣ��\n Python���������ҵ΢��API,���Խ�Ӧ�á��澯����Ӧ�á����͸�����Ա����Ϣ��")
    # wx.send_data("���ǳ����͵ĵ�2����Ϣ��")
    # wx.send_data("���ǳ����͵ĵ�2����Ϣ��")

