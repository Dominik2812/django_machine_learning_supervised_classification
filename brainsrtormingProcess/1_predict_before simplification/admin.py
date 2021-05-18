from django.contrib import admin
from .models import BaseData, CrossVal, OptimizedHyperParameters, MajorityVote

admin.site.register(BaseData)
admin.site.register(CrossVal)
admin.site.register(OptimizedHyperParameters)
admin.site.register(MajorityVote)





