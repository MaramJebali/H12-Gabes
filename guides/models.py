from django.db import models


class Guide(models.Model):
    SEX_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    LOCATION_CHOICES = [
        ('Souk Jara', 'Souk Jara'),
        ('Chenini Oasis', 'Chenini Oasis'),
        ('Plage de la Corniche', 'Plage de la Corniche'),
        ('Sidi Abolbabah Ansari Mosque', 'Sidi Abolbabah Ansari Mosque'),
        ('Musée ethnographique de Gabès', 'Musée ethnographique de Gabès'),
        ("L'église de Gabès", "L'église de Gabès"),
        ('Mosque of Sidi Driss', 'Mosque of Sidi Driss'),
        ('Site De Zraoua Ancienne', 'Site De Zraoua Ancienne'),
        ('Tamazret, Matmata', 'Tamazret, Matmata'),
        ('El Hamma-Gabès', 'El Hamma-Gabès'),
        ('El Mdou Gabès', 'El Mdou Gabès'),
        ('Toujane, Matmata', 'Toujane, Matmata'),
    ]

    name = models.CharField(max_length=255)
    cin = models.CharField(max_length=8, unique=True)
    phone_number = models.CharField(max_length=15)
    picture = models.ImageField(upload_to='guide_pics/')
    age = models.PositiveIntegerField()
    sex = models.CharField(max_length=1, choices=SEX_CHOICES)
    living_place = models.CharField(max_length=255)
    locations = models.TextField()  # Comma-separated location values

    def __str__(self):
        return self.name


class Reservation(models.Model):
    tourist_name = models.CharField(max_length=255)
    tourist_phone_number = models.CharField(max_length=15)
    guide = models.ForeignKey(Guide, on_delete=models.CASCADE, related_name='reservations')
    tour_date = models.DateField()
    message = models.TextField(blank=True)

    def __str__(self):
        return f"Reservation for {self.tourist_name} with {self.guide.name} on {self.tour_date}"