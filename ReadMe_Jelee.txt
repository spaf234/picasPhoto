--------- python 환경구성 -----------

01_python install : 설치위치 지정은  : C:\PICAS\Python\Python312
02 파이썬 가상환경 생성 : 
      cd C:\PICAS\Python\Python312   
      python -m venv venvPicas

03. venvPicas 압축해제후 overwrite


-------- main_faceDetect.py 실행 ----------------
01:  cmd창 두개 실행 
        cd C:\PICAS\Python\Python312\venvPicas\Scripts
        activate.bat 실행하여 프롬프트 venvPicas 변경확인

02: 첫번째 cmd창 ( faseAPI REST 웹 서버 실행) : 

                       python main_faceDetect.py

                       확인 : chrome, edge :  http://localhost:8000/docs
                       ( try it out 시 이미지를 base64로 입력설계여서 아래 testUI를 만듬)



03: 두번째 cmd창( faseAPI 서버 테스트용 UI)  :  
                      
                       python gradio_face_detect.py

                       확인 : chrome, edge :  http://localhost:7860



04:  테스트 웹 자바스크립트  face_detect_js_test.html
04:  테스트 웹 자바스크립트+ pyscript    face_detect_pyscirpt_test.html


05:  Batch 품질 얼굴크기 자바스크립트 :  passport_photo_js_test.html
05:  Batch 품질 얼굴크기 자바스크립트 + pyscript  :  passport_photo_pyscript_test.html
05:  static폴더에 js offline용 저장됨


09:참조 ( c# winform 테스트 UI ) :
 
 C:\PICAS\Python\Python312\venvPicas\Scripts\cSharp\bin\Debug\net48\csharp.exe 

==========SAM SegmentAnything=============================================
activate.bat 후에 site-package를 공유 활성화후에 

sam_ui\run_ui.bat
sam2_ui\run_ui.bat ( video)


========== SAM with CLIP prompt =====================
activate.bat 후에 site-package를 공유 활성화후에 

cd C:\PICAS\Python\Python312\venvPicas\Scripts\sam_clip_ui\segment-anything-with-clip
gradio app.py 로 실행

SAM모델이 2GB로 크다




=================pip freeze 로 윈도우에서 linux용으로 만들기 =======================================


윈도우에서 만든 requirements.txt 를 리눅스에선 사용할 수 없었습니다. (OSError)

"pip freeze > requirements.txt" 대신 "pip list --format=freeze > requirements.txt"를 사용해야 합니다.




