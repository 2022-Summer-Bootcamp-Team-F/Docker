# Create your tasks here
# from __future__ import absolute_import, unicode_literals

from celery import shared_task
import sys, os

# app과 같은 선상에 있는 DualStyleGAN import 할 수 있게 절대 경로 참조 코드
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


@shared_task
def ai_transfer(input_url, 화풍 id):
    1. client에서 보낸 input 이미지 가져오기
    2. input 이미지를 ai 코드 경로에 넣어주기
    3. 화풍 선택 id를 가져와서 style 선택해주기
    4. 결과 이미지를 s3에 업로드 하고 그 url을 가져와서 client에 전달해주기
