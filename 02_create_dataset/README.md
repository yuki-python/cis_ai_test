https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
上記からモデルをダウンロード
このモデルは Google が提供する MediaPipe Solutions の一部(Apache License 2.0)



実行順番
①merge_raw_data.py
　→複数デバイスのesp32-s3、そして何セットか取っているデータをprefixをつけて一つのデータセットとしてまとめるマージプログラムを実行
②syncro_timestamp.py
　→タイムスタンプでカメラとcsiデータの紐づけをする
③anotation.py
　→mediapipeライブラリを使った骨格推定を行う自動プログラム。
　→これで骨格点に関する情報をlabelme形式で自動吐出し。
④export_synced_dataset.py
　→以上によって作られた正解ラベル付きのデータセットをモデル学習用フォルダに丸ごと移動させる。



This project uses MediaPipe Pose Landmarker model licensed under Apache License 2.0.