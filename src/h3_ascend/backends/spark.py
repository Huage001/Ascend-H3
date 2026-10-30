class SparkAttention:
    def __init__(self, layout, **kwargs):
        import os

        from h3_ascend._vendor.spark import settings

        os.environ["SPARK_QUERY_GROUP"] = str(settings.get("SPARK_QUERY_GROUP", "1024"))
        from h3_ascend._vendor.spark.backend import Spark

        self.layout = layout
        self.impl = Spark(layout.spark_case(), solver="panel")

    def __call__(self, q, k, v, **kwargs):
        from h3_ascend._vendor.spark import settings

        token = settings.set_frames(self.layout.frames)
        try:
            return self.impl(q[None], k[None], v[None])[0]
        finally:
            settings._frames.reset(token)
