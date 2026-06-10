#include <string.h>
#include <inttypes.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_timer.h"

#define WIFI_SSID      "elecom-8a0697"
#define WIFI_PASS      "mna5u2wmwnkm"

static void wifi_event_handler(void *arg,
                               esp_event_base_t event_base,
                               int32_t event_id,
                               void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    }
}

static void csi_callback(void *ctx, wifi_csi_info_t *info)
{
    uint64_t ts = esp_timer_get_time();

    printf("CSI,ts=%" PRIu64 ",len=%d,rssi=%d,mac=%02X:%02X:%02X:%02X:%02X:%02X,buf=",
           ts,
           info->len,
           info->rx_ctrl.rssi,
           info->mac[0], info->mac[1], info->mac[2],
           info->mac[3], info->mac[4], info->mac[5]);

    for (int i = 0; i < info->len; i++) {
        printf("%d", (int8_t)info->buf[i]);
        if (i != info->len - 1) printf(",");
    }
    printf("\n");
}

void app_main(void)
{
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL);

    esp_wifi_set_ps(WIFI_PS_NONE);

    wifi_config_t wifi_config = {0};
    strcpy((char *)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char *)wifi_config.sta.password, WIFI_PASS);

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    wifi_promiscuous_filter_t filter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_DATA
    };
    esp_wifi_set_promiscuous_filter(&filter);
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi(true);
    esp_wifi_set_csi_rx_cb(csi_callback, NULL);
}
