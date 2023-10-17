import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.config.experimental.list_physical_devices('GPU'):
    for device in tf.config.experimental.list_physical_devices('GPU'):
        print(f"Details for {device.device_type} {device.name}:")
        print("   - Memory size:", device.memory_limit // (1024 ** 2), "MB")
        print("   - Device ID:", device.device_type.lower().replace(" ", "_"))
