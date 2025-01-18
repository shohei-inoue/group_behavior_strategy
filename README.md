# group behavior strategy

![frame_2999](https://github.com/user-attachments/assets/2ed8cc46-554e-4b1b-ae70-a7a0007ae89f)

## プロジェクト構成

- `src/main.py`: アプリケーションのエントリーポイント。
- `envs/group_behavior_strategy_env.py`: 強化学習環境を定義するファイル。
- `agents/red_agents.py`: 強化学習エージェントを定義するファイル。
- `robots/red.py`: ロボットの動作を定義するファイル。
- `Dockerfile`: Dockerイメージを構築するための設定ファイル。Python 3.12.1をベースに、NumPy、Gym、Matplotlibをインストールします。
- `docker-compose.yml`: Docker Composeの設定ファイル。サービスの定義やボリュームの設定を行い、コンテナを簡単に立ち上げることができます。
- `.env`: データベース接続情報などの環境変数を定義するファイル。
- `requirements.txt`: プロジェクトで使用するPythonパッケージのリスト。

## セットアップ手順

1. このリポジトリをクローンします。
   ```sh
   git clone <repository-url>
2. DockerとDocker Composeがインストールされていることを確認します。
3. .envファイルをプロジェクトのルートディレクトリに作成し, データベース接続情報を記述します。
```
DATABASE_URL=postgresql://user:password@localhost:5432/mydatabase
```
4. Docker Composeを使用してコンテナをビルドし、起動します。
```
docker-compose up --build
```

## actor critic
Actor-Critic モデル設計:
Actor: 行動ポリシーを学習し、環境の観測からアクションをサンプリング。
Critic: 状態価値関数 𝑉(𝑠)を学習。

損失関数:
Actor の損失: \[log(𝜋(𝑎∣𝑠))⋅𝐴(𝑠,𝑎)−log(π(a∣s))⋅A(s,a)\]（ポリシー勾配法）。
Critic の損失: MSE(𝑅,𝑉(𝑠))
エントロピー正則化（探索性維持）。
トレーニングループ:
環境からデータを収集し、Actor と Critic のネットワークを更新。

## やること
- 画像やgifの保存場所の変更
- 走行可能性の初期分布の低さの調整
- 報酬の再設計
