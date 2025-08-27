from data_loader import GliderDataLoader, GulfStreamLoader, MPADataLoader, MapDataLoader
from nessie_interpolation_function import create_spatial_interpolation
import datetime

loader = MapDataLoader()
df = loader.load_data()


fig, metadata = create_spatial_interpolation(
    df=df,
    parameter='rhodamine',
    hours_back=3,
    nan_filter_parameters=['pHin','rhodamine'],
    platform_filter='Glider',
    layer_filter='MLD'
)

fig.show()