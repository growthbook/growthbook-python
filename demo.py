import asyncio

from aiohttp import payload_type

from growthbook import GrowthBookClient, Options, UserContext, Experiment, GrowthBook


def my_tracking_callback(experiment, result, user):
    print(f"📊 Tracking: {experiment.key}, variation={result.variationId}, user={user.attributes}")

async def main():
    gb = GrowthBook(
        attributes={"id": 1},
        trackingCallback=my_tracking_callback,
        api_host="https://6526d8a46e76.ngrok-free.app",
        client_key="sdk-HS7oAdaI8Yi4Bh"
    )
    gb.eval_feature()
    # 1. Ініціалізація GrowthBook
    client = GrowthBookClient(
        Options(
            api_host="https://6526d8a46e76.ngrok-free.app",
            client_key="sdk-HS7oAdaI8Yi4Bh",
            # заміни на свій client key,
            remote_eval = True,
            global_attributes={"id": 1},
        )
    )

    try:
        # 2. Завантажуємо фічі
        success = await client.initialize()
        if not success:
            print("❌ Не вдалося ініціалізувати GrowthBook клієнт")
            return
        print("✅ GrowthBook клієнт готовий")

        # 3. Створюємо користувача
        user = UserContext(
            attributes={
                "id": "user_123",
                "country": "US",
                "premium": True
            }
        )

        # 4. Перевіряємо просту фічу
        if await client.is_on("new-boolean-feature-september", user):
            print("🟢 Нова домашня сторінка увімкнена!")
        else:
            print("🔴 Нова домашня сторінка вимкнена!")

        # 5. Дістаємо значення фічі з fallback
        color = await client.get_feature_value("new-boolean-feature-september", True, user)
        print(f"🎨 Колір кнопки: {color}")

        # 6. Запускаємо експеримент
        experiment = Experiment(
            key="pricing-test",
            variations=["$9.99", "$14.99", "$19.99"]
        )

        result = await client.run(experiment, user)
        if result.inExperiment:
            print(f"🧪 Користувач у експерименті: {result.value}")
        else:
            print("⚪ Користувач не включений в експеримент")

    finally:
        # 7. Закриваємо клієнт
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
