# Method native dipanggil lewat JNI berdasarkan nama -> jangan di-obfuscate.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}
-keep class com.crosschick.freshness.ml.FreshnessNative { *; }
