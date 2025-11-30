#!/usr/bin/env python3
"""Тестовый скрипт для проверки поиска локаций со всеми параметрами."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from src.api.openaq import OpenAQClient
from src.data.models import CITIES
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def test_location_search():
    """Тестирует поиск локаций для каждого города."""
    client = OpenAQClient()
    
    print("=" * 80)
    print("ТЕСТ: Поиск локаций со всеми параметрами для каждого города")
    print("=" * 80)
    print()
    
    for city in CITIES:
        print(f"\n{'='*80}")
        print(f"Город: {city.name} ({city.country})")
        print(f"Координаты: {city.latitude}, {city.longitude}")
        print(f"{'='*80}")
        
        try:
            # Получаем локации по координатам
            print(f"\n1. Поиск локаций по координатам (радиус 25км)...")
            locations = client.get_locations(
                latitude=city.latitude,
                longitude=city.longitude,
                radius=25000,
                limit=30
            )
            
            if not locations:
                print("   ❌ Локации не найдены")
                continue
            
            print(f"   ✅ Найдено локаций: {len(locations)}")
            
            # Анализируем параметры в каждой локации
            print(f"\n2. Анализ параметров в локациях...")
            location_param_map = {}
            for loc in locations:
                loc_id = loc.get("id")
                if not loc_id:
                    continue
                sensors = loc.get("sensors", [])
                loc_params = set()
                for sensor in sensors:
                    param_info = sensor.get("parameter", {})
                    if isinstance(param_info, dict):
                        param_name = param_info.get("name", "").lower()
                        if param_name in ["pm25", "pm10", "no2", "o3"]:
                            loc_params.add(param_name)
                if loc_params:
                    location_param_map[loc_id] = {
                        "name": loc.get("name", ""),
                        "params": loc_params
                    }
            
            print(f"   ✅ Локаций с нужными параметрами: {len(location_param_map)}")
            
            # Показываем первые 5 локаций с параметрами
            print(f"\n   Первые 5 локаций:")
            for i, (loc_id, info) in enumerate(list(location_param_map.items())[:5], 1):
                print(f"   {i}. ID {loc_id}: {info['name'][:50]}")
                print(f"      Параметры: {sorted(info['params'])}")
            
            # Ищем локацию со всеми параметрами
            required_params = {"pm25", "pm10", "no2", "o3"}
            print(f"\n3. Поиск локации со ВСЕМИ параметрами...")
            all_4_locs = []
            for loc_id, info in location_param_map.items():
                if info["params"] >= required_params:
                    all_4_locs.append((loc_id, info))
            
            if all_4_locs:
                print(f"   ✅ Найдена локация со ВСЕМИ параметрами!")
                loc_id, info = all_4_locs[0]
                print(f"   ID: {loc_id}")
                print(f"   Название: {info['name']}")
                print(f"   Параметры: {sorted(info['params'])}")
            else:
                print(f"   ⚠️  Нет локации со всеми параметрами")
                
                # Показываем как можно скомбинировать
                print(f"\n4. Комбинирование локаций для покрытия всех параметров...")
                covered = set()
                selected = []
                remaining = required_params.copy()
                
                # Сортируем по количеству нужных параметров
                sorted_locs = sorted(
                    location_param_map.items(),
                    key=lambda x: len(x[1]["params"] & remaining),
                    reverse=True
                )
                
                for loc_id, info in sorted_locs:
                    if remaining:
                        new_params = info["params"] & remaining
                        if new_params:
                            selected.append((loc_id, info["name"], new_params))
                            covered.update(new_params)
                            remaining -= info["params"]
                            print(f"   - ID {loc_id}: {info['name'][:40]}")
                            print(f"     Добавляет параметры: {sorted(new_params)}")
                            if not remaining:
                                break
                
                if covered >= required_params:
                    print(f"\n   ✅ Можно собрать все параметры из {len(selected)} локаций:")
                    for loc_id, name, params in selected:
                        print(f"      • {name[:50]} - {sorted(params)}")
                else:
                    missing = required_params - covered
                    print(f"\n   ❌ Не хватает параметров: {sorted(missing)}")
                    print(f"   Доступные параметры: {sorted(covered)}")
                    
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("ТЕСТ ЗАВЕРШЕН")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_location_search()

