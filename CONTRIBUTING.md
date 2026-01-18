# Contributing to Crypto Quant Signals Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Git

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/crypto-quant-signals-platform.git
cd crypto-quant-signals-platform

# Add upstream remote
git remote add upstream https://github.com/aban369/crypto-quant-signals-platform.git
```

### Setup Development Environment

```bash
# Run quick start script
chmod +x scripts/quick_start.sh
./scripts/quick_start.sh

# Or manual setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

cd ../frontend
npm install
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

### 2. Make Changes

- Write clean, readable code
- Follow coding standards (see below)
- Add tests for new features
- Update documentation

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add new signal aggregation method"
```

Commit message format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Coding Standards

### Python (Backend)

Follow PEP 8 style guide:

```python
# Good
def calculate_ofi(orderbook: Dict, level: int) -> float:
    """
    Calculate Order Flow Imbalance at specified level.
    
    Args:
        orderbook: Order book snapshot
        level: Price level
        
    Returns:
        OFI value
    """
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    # Calculate imbalance
    ofi = sum([b[1] for b in bids[:level]]) - sum([a[1] for a in asks[:level]])
    
    return float(ofi)
```

**Tools**:
```bash
# Format code
black backend/

# Lint
flake8 backend/

# Type checking
mypy backend/
```

### TypeScript/React (Frontend)

Follow Airbnb style guide:

```typescript
// Good
interface SignalCardProps {
  symbol: string;
  signal: TradingSignal;
  onSelect?: (symbol: string) => void;
}

const SignalCard: React.FC<SignalCardProps> = ({ symbol, signal, onSelect }) => {
  const handleClick = () => {
    onSelect?.(symbol);
  };

  return (
    <div className="signal-card" onClick={handleClick}>
      <h3>{symbol}</h3>
      <p>Direction: {signal.direction}</p>
    </div>
  );
};

export default SignalCard;
```

**Tools**:
```bash
# Format code
npm run format

# Lint
npm run lint

# Type check
npm run type-check
```

---

## Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test
pytest tests/test_hawkes.py::test_flash_crash_detection
```

**Test Structure**:
```python
import pytest
from core.hawkes.flash_crash_detector import FlashCrashDetector

def test_flash_crash_detection():
    """Test flash crash detection with synthetic data"""
    detector = FlashCrashDetector()
    
    # Create test data
    prices = np.array([100, 95, 90, 85, 95, 100])
    timestamps = np.arange(len(prices)) * 1000
    volumes = np.ones(len(prices)) * 100
    
    # Detect crashes
    crashes = detector.detect_flash_crash(prices, timestamps, volumes)
    
    # Assertions
    assert len(crashes) > 0
    assert crashes[0].classification in ["MINOR", "MODERATE", "SEVERE", "EXTREME"]
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test
npm test SignalCard.test.tsx
```

**Test Structure**:
```typescript
import { render, screen } from '@testing-library/react';
import SignalCard from './SignalCard';

describe('SignalCard', () => {
  it('renders signal information correctly', () => {
    const mockSignal = {
      direction: 'LONG',
      confidence: 0.85,
      // ... other properties
    };

    render(<SignalCard symbol="BTC/USDT" signal={mockSignal} />);

    expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
    expect(screen.getByText('LONG')).toBeInTheDocument();
  });
});
```

---

## Documentation

### Code Documentation

**Python**:
```python
def calculate_hawkes_intensity(
    t: float,
    past_events: np.ndarray,
    mu: float,
    alpha: np.ndarray,
    beta: np.ndarray
) -> float:
    """
    Calculate Hawkes process intensity at time t.
    
    The intensity function is:
    λ(t) = μ + Σ Σ α_j * exp(-β_j * (t - t_i))
    
    Args:
        t: Current time
        past_events: Array of past event times
        mu: Background intensity
        alpha: Excitation coefficients
        beta: Decay rates
        
    Returns:
        Intensity value at time t
        
    Example:
        >>> intensity = calculate_hawkes_intensity(
        ...     t=100.0,
        ...     past_events=np.array([90.0, 95.0]),
        ...     mu=0.1,
        ...     alpha=np.array([0.5]),
        ...     beta=np.array([1.0])
        ... )
        >>> print(f"Intensity: {intensity:.4f}")
    """
    # Implementation
    pass
```

**TypeScript**:
```typescript
/**
 * Fetch trading signal for a symbol
 * 
 * @param symbol - Trading symbol (e.g., "BTC/USDT")
 * @returns Promise resolving to TradingSignal
 * 
 * @example
 * ```typescript
 * const signal = await fetchSignal('BTC/USDT');
 * console.log(signal.direction); // "LONG" | "SHORT" | "NEUTRAL"
 * ```
 */
export async function fetchSignal(symbol: string): Promise<TradingSignal> {
  // Implementation
}
```

### README Updates

When adding new features, update:
- Main README.md
- Relevant documentation in docs/
- API documentation if endpoints change

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
Add screenshots for UI changes
```

### Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainers
3. Address feedback
4. Approval and merge

---

## Areas for Contribution

### High Priority

- [ ] Real-time data integration (Binance, Bybit WebSockets)
- [ ] Model training pipelines
- [ ] Backtesting framework
- [ ] Performance optimization
- [ ] Additional exchange support

### Medium Priority

- [ ] Mobile app (React Native)
- [ ] Advanced charting
- [ ] Alert system
- [ ] Portfolio tracking
- [ ] Social features

### Documentation

- [ ] Tutorial videos
- [ ] Example strategies
- [ ] Research paper summaries
- [ ] API client libraries
- [ ] Deployment guides

---

## Research Contributions

We welcome implementations of new research papers!

### Process

1. Open an issue proposing the paper
2. Discuss implementation approach
3. Create implementation in `backend/core/`
4. Add tests and documentation
5. Submit PR

### Requirements

- Paper must be peer-reviewed or from reputable source
- Implementation must be well-documented
- Include mathematical formulas in docstrings
- Add to RESEARCH_PAPERS.md

---

## Community

### Communication Channels

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Email: raiz.s.group1@gmail.com

### Getting Help

- Check existing issues and documentation
- Ask in GitHub Discussions
- Reach out to maintainers

---

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Crypto Quant Signals Platform! 🚀
