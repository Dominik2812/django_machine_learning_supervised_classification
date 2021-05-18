# Generated by Django 3.1.3 on 2021-04-13 05:15

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0007_crossvalscore_meanscore_singlescore'),
    ]

    operations = [
        migrations.CreateModel(
            name='Hello',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(default='Hello', max_length=30)),
                ('data', models.CharField(default='Hello', max_length=30)),
                ('baseData', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Hello', to='predict.basedata')),
            ],
        ),
    ]
