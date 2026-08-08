"""Keyless Azure Retail Prices client.

``https://prices.azure.com/api/retail/prices`` is unauthenticated, so the overage side of
the newsvendor ratio can be a live, cited, per-SKU, per-region number instead of a constant
somebody picked. That is what turns the newsvendor framing from a thought experiment into a
system, and it costs nothing — which the zero-cost rule requires.

Two honesty obligations travel with the number and are carried in the returned record:

* **Retail list price is not what anyone actually pays.** Reservations, savings plans, spot
  and enterprise agreements all move it. ``discount_multiplier`` exists so the *sensitivity*
  can be shown, because the sensitivity is the interesting output, not the absolute figure.
* **A price has an as-of time.** ``retrieved_at`` is recorded and surfaced, never implied.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from delphi.timeutil import utc_now

RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"
API_VERSION = "2023-01-01-preview"
SOURCE_ID = "azure-retail-prices"


@dataclass(frozen=True)
class RetailPrice:
    """One retail price observation, with everything needed to cite it."""

    sku: str
    product: str
    region: str
    unit: str
    price: float
    currency: str
    retrieved_at: datetime
    #: 1.0 = list price. Lower values model a negotiated or reserved rate.
    discount_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("a retail price cannot be negative")
        if not 0 < self.discount_multiplier <= 1:
            raise ValueError("discount_multiplier must lie within (0, 1]")

    @property
    def effective_price(self) -> float:
        """List price after the declared discount assumption."""
        return self.price * self.discount_multiplier

    @property
    def is_list_price(self) -> bool:
        return self.discount_multiplier == 1.0

    def with_discount(self, multiplier: float) -> "RetailPrice":
        return RetailPrice(
            sku=self.sku,
            product=self.product,
            region=self.region,
            unit=self.unit,
            price=self.price,
            currency=self.currency,
            retrieved_at=self.retrieved_at,
            discount_multiplier=multiplier,
        )


def parse_price_items(payload: dict[str, Any], *, retrieved_at: datetime) -> list[RetailPrice]:
    """Defensively parse the API's ``Items`` array.

    Rows missing a usable price, unit or region are skipped rather than defaulted: a
    silently zero-priced SKU would drive the newsvendor ratio to buy infinite capacity.
    """
    items = payload.get("Items")
    if not isinstance(items, list):
        raise ValueError("Azure retail price payload has no Items array")
    prices: list[RetailPrice] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_price = item.get("retailPrice")
        region = item.get("armRegionName") or ""
        unit = item.get("unitOfMeasure") or ""
        if not isinstance(raw_price, int | float) or raw_price <= 0 or not region or not unit:
            continue
        prices.append(
            RetailPrice(
                sku=str(item.get("skuName") or item.get("armSkuName") or "unknown"),
                product=str(item.get("productName") or ""),
                region=str(region),
                unit=str(unit),
                price=float(raw_price),
                currency=str(item.get("currencyCode") or "USD"),
                retrieved_at=retrieved_at,
            )
        )
    return prices


def _odata_literal(value: str) -> str:
    """Quote a string for an OData filter, escaping embedded single quotes by doubling.

    These values come from config rather than from a request, so this is not a live
    injection path — but an unescaped apostrophe would silently malform the query and
    return the wrong SKU's price, which then propagates into the newsvendor ratio as a
    plausible-looking number. Cheaper to escape than to debug.
    """
    return "'" + value.replace("'", "''") + "'"


def build_filter(*, region: str, service_name: str, sku_contains: str | None = None) -> str:
    """Compose the OData ``$filter`` the API expects."""
    clauses = [
        f"armRegionName eq {_odata_literal(region)}",
        f"serviceName eq {_odata_literal(service_name)}",
    ]
    if sku_contains:
        clauses.append(f"contains(skuName, {_odata_literal(sku_contains)})")
    return " and ".join(clauses)


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def fetch_retail_prices(
    *,
    region: str = "southeastasia",
    service_name: str = "Virtual Machines",
    sku_contains: str | None = None,
    limit: int | None = None,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> list[RetailPrice]:
    """Fetch current retail prices. No API key, no account, no cost.

    ``limit`` truncates **client-side**. Verified against the live endpoint on 2026-08-03:
    the API ignores OData ``$top`` and returns a fixed page of up to 1000 items with a
    ``NextPageLink`` for the rest. Sending ``$top`` would imply a server-side bound the
    service does not honour, so it is not sent. Only the first page is read — enough for
    picking a representative SKU price, and it keeps one call to one free endpoint.
    """
    params = {
        "api-version": API_VERSION,
        "$filter": build_filter(
            region=region, service_name=service_name, sku_contains=sku_contains
        ),
    }
    retrieved_at = utc_now()
    owned = client is None
    session = client or httpx.Client(timeout=timeout)
    try:
        response = session.get(RETAIL_PRICES_URL, params=params)
        response.raise_for_status()
        prices = parse_price_items(response.json(), retrieved_at=retrieved_at)
    finally:
        if owned:
            session.close()
    return prices if limit is None else prices[:limit]


def cheapest_hourly(prices: list[RetailPrice]) -> RetailPrice:
    """Pick the cheapest per-hour SKU, which is the natural unit for a replica-hour.

    Non-hourly units (per-month reservations, per-GB) are excluded rather than converted:
    a monthly reservation price is a different commitment, and silently dividing it by 730
    would smuggle a purchasing decision into a units conversion.
    """
    hourly = [price for price in prices if "hour" in price.unit.lower()]
    if not hourly:
        raise ValueError("no hourly-priced SKU in the supplied prices")
    return min(hourly, key=lambda price: (price.effective_price, price.sku))
