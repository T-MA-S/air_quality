#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для переобучения ML-моделей с улучшенными признаками."""

import sys
sys.path.insert(0, '/opt/airflow/src')

from train_ml_model import main

if __name__ == "__main__":
    print("=" * 80)
    print("Переобучение ML-моделей с улучшенными признаками")
    print("=" * 80)
    print("\nУлучшения:")
    print("  - Добавлены лаги PM2.5 (1h, 2h, 3h, 6h, 24h)")
    print("  - Добавлены скользящие средние (3h, 6h, 24h)")
    print("  - Добавлены временные признаки (час, день недели, месяц)")
    print("  - Добавлены циклические признаки времени")
    print("  - Добавлены взаимодействия признаков")
    print("  - Улучшены гиперпараметры модели")
    print("  - Увеличено количество дней для обучения (90 вместо 60)")
    print("=" * 80)
    print()
    
    results = main()
    
    print("\n" + "=" * 80)
    print("Переобучение завершено!")
    print("=" * 80)
    print(f"\nОбучено моделей: {len(results)}")
    for result in results:
        if result:
            print(f"\n{result['city']}:")
            print(f"  Test R²: {result['metrics']['test_r2']:.3f}")
            print(f"  Test RMSE: {result['metrics']['test_rmse']:.3f}")
            if result.get('backtest'):
                print(f"  Backtest R²: {result['backtest']['overall_r2']:.3f}")
                print(f"  Backtest RMSE: {result['backtest']['overall_rmse']:.3f}")


