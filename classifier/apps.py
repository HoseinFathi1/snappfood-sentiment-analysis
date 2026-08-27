from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    def ready(self):
        # Loading the model here (once, when the Django server starts) means
        # every request just runs inference — it doesn't reload the model
        # from disk each time, which would be slow.
        from . import model_loader
        model_loader.load_model()
