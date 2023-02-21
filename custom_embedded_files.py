import fiftyone.core.fields as fof
from fiftyone.core.odm import DynamicEmbeddedDocument


class BoatBusDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing boatbus data about generic samples."""

    date = fof.DateTimeField()
    time = fof.DateTimeField()
    rateOfTurn = fof.FloatField()
    heading = fof.FloatField()
    roll = fof.FloatField()
    yaw = fof.FloatField()
    pitch = fof.FloatField()
    courseOverGround = fof.FloatField()
    speedOverGround = fof.FloatField()
    speedOverWater = fof.FloatField()
    windAngle = fof.FloatField()
    waterTemperature = fof.FloatField()
    windSpeed = fof.FloatField()
    longitude = fof.FloatField()
    latitude = fof.FloatField()


class ImuDataEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing boatbus data about generic samples."""

    roll = fof.FloatField()
    pitch = fof.FloatField()
    yaw = fof.FloatField()
    aPerpendicular = fof.FloatField()
    aLeteral = fof.FloatField()
    aLongitundinal = fof.FloatField()


class SensorsEmbeddedDocument(DynamicEmbeddedDocument):
    """Base class for storing sensors.json data about generic samples."""

    version = fof.StringField()
    UUID = fof.StringField()

    @classmethod
    def build_from(cls, sensors_version, sensors_UUID):
        return cls._build_from_dict(sensors_version, sensors_UUID)

    @classmethod
    def _build_from_dict(cls, sensors_version, sensors_UUID):
        version = sensors_version
        UUID = sensors_UUID

        return cls(
            version=version,
            UUID=UUID
        )
