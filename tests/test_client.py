

from growthbook.common_types import Experiment, Options, UserContext
from growthbook.growthbook_client import GrowthBookClient

def on_experiment_viewed(experiment, result):
    print(f"Experiment viewed: {experiment.key}, {result.variationId}")

client = GrowthBookClient(
        options=Options(
            api_host="https://cdn.growthbook.io",
            client_key="sdk-aBoRSj7NifTJcsXD",
            on_experiment_viewed=on_experiment_viewed,
            enabled=True,
        )
    )

client.initialize()

user_context = UserContext(attributes={"id": "1"})

result = client.run(Experiment(key="test", variations=["A", "B"]), user_context)

print(result)