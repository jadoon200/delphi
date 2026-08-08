"""Keyless Azure Retail Prices client, mocked with respx per the ingester convention."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from delphi.data.prices import (
    RETAIL_PRICES_URL,
    RetailPrice,
    build_filter,
    cheapest_hourly,
    fetch_retail_prices,
    parse_price_items,
)

RETRIEVED = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

PAYLOAD = {
    "Items": [
        {
            "skuName": "B2s",
            "productName": "Virtual Machines Bs Series",
            "armRegionName": "southeastasia",
            "unitOfMeasure": "1 Hour",
            "retailPrice": 0.0416,
            "currencyCode": "USD",
        },
        {
            "skuName": "D4s v5",
            "productName": "Virtual Machines Dsv5 Series",
            "armRegionName": "southeastasia",
            "unitOfMeasure": "1 Hour",
            "retailPrice": 0.192,
            "currencyCode": "USD",
        },
        {
            "skuName": "Reserved",
            "productName": "Virtual Machines",
            "armRegionName": "southeastasia",
            "unitOfMeasure": "1 Month",
            "retailPrice": 60.0,
            "currencyCode": "USD",
        },
    ]
}


def test_parser_skips_rows_that_would_poison_the_cost_ratio() -> None:
    """A zero or missing price must be dropped, never defaulted.

    A silently zero-priced SKU drives the newsvendor ratio to buy unbounded capacity.
    """
    payload = {
        "Items": [
            {
                "skuName": "ok",
                "armRegionName": "eastus",
                "unitOfMeasure": "1 Hour",
                "retailPrice": 0.1,
                "currencyCode": "USD",
            },
            {
                "skuName": "zero",
                "armRegionName": "eastus",
                "unitOfMeasure": "1 Hour",
                "retailPrice": 0.0,
            },
            {"skuName": "no-region", "unitOfMeasure": "1 Hour", "retailPrice": 0.5},
            {"skuName": "no-unit", "armRegionName": "eastus", "retailPrice": 0.5},
            "not-a-dict",
        ]
    }
    parsed = parse_price_items(payload, retrieved_at=RETRIEVED)
    assert [price.sku for price in parsed] == ["ok"]


def test_parser_rejects_a_payload_with_no_items() -> None:
    with pytest.raises(ValueError, match="no Items array"):
        parse_price_items({}, retrieved_at=RETRIEVED)


def test_filter_composes_valid_odata() -> None:
    clause = build_filter(region="southeastasia", service_name="Virtual Machines")
    assert clause == "armRegionName eq 'southeastasia' and serviceName eq 'Virtual Machines'"
    assert "contains(skuName, 'D4')" in build_filter(
        region="eastus", service_name="Virtual Machines", sku_contains="D4"
    )


@respx.mock
def test_fetch_requires_no_authentication_and_records_retrieval_time() -> None:
    route = respx.get(RETAIL_PRICES_URL).mock(return_value=httpx.Response(200, json=PAYLOAD))
    prices = fetch_retail_prices(region="southeastasia")
    assert route.called
    assert "authorization" not in {key.lower() for key in route.calls[0].request.headers}
    assert len(prices) == 3
    assert all(price.retrieved_at.tzinfo is not None for price in prices)


@respx.mock
def test_fetch_retries_a_transient_failure_then_succeeds() -> None:
    respx.get(RETAIL_PRICES_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=PAYLOAD),
        ]
    )
    assert len(fetch_retail_prices()) == 3


def test_cheapest_hourly_ignores_non_hourly_commitments() -> None:
    """A monthly reservation is a different purchase, not a cheaper hour."""
    prices = parse_price_items(PAYLOAD, retrieved_at=RETRIEVED)
    cheapest = cheapest_hourly(prices)
    assert cheapest.sku == "B2s"
    assert "hour" in cheapest.unit.lower()


def test_cheapest_hourly_raises_when_nothing_is_priced_per_hour() -> None:
    monthly = [
        RetailPrice(
            sku="m",
            product="p",
            region="eastus",
            unit="1 Month",
            price=10.0,
            currency="USD",
            retrieved_at=RETRIEVED,
        )
    ]
    with pytest.raises(ValueError, match="no hourly-priced SKU"):
        cheapest_hourly(monthly)


def test_discount_multiplier_models_negotiated_rates_without_hiding_list_price() -> None:
    price = RetailPrice(
        sku="D4s v5",
        product="p",
        region="southeastasia",
        unit="1 Hour",
        price=0.20,
        currency="USD",
        retrieved_at=RETRIEVED,
    )
    assert price.is_list_price
    discounted = price.with_discount(0.6)
    assert discounted.effective_price == pytest.approx(0.12)
    assert discounted.price == pytest.approx(0.20), "list price must remain visible"
    assert not discounted.is_list_price


def test_impossible_discounts_are_refused() -> None:
    with pytest.raises(ValueError, match="discount_multiplier"):
        RetailPrice(
            sku="s",
            product="p",
            region="r",
            unit="1 Hour",
            price=1.0,
            currency="USD",
            retrieved_at=RETRIEVED,
            discount_multiplier=1.5,
        )


@respx.mock
def test_top_is_not_sent_because_the_api_ignores_it() -> None:
    """Verified live 2026-08-03: the endpoint returns a full page regardless of $top.

    Sending it would imply a server-side bound the service does not honour.
    """
    route = respx.get(RETAIL_PRICES_URL).mock(return_value=httpx.Response(200, json=PAYLOAD))
    fetch_retail_prices(limit=1)
    assert "$top" not in str(route.calls[0].request.url)


@respx.mock
def test_limit_truncates_client_side() -> None:
    respx.get(RETAIL_PRICES_URL).mock(return_value=httpx.Response(200, json=PAYLOAD))
    assert len(fetch_retail_prices(limit=2)) == 2
    assert len(fetch_retail_prices()) == 3
