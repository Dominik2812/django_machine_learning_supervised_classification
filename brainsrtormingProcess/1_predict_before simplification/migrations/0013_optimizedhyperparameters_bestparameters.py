# Generated by Django 3.1.3 on 2021-04-13 15:21

from django.db import migrations
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0012_auto_20210413_1003'),
    ]

    operations = [
        migrations.AddField(
            model_name='optimizedhyperparameters',
            name='bestParameters',
            field=picklefield.fields.PickledObjectField(default='OptimizedHyperParameters', editable=False),
        ),
    ]