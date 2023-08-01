from django.shortcuts import render

class TemplateRender:
    def index(request):
        return render(request,"index.html")
    def Text2Image(request):
        return render(request,"text2img.html")
    def Image2Image(request):
        return render(request,"img2img.html")
    def Text2Video(request):
        return render(request,"text2video.html")